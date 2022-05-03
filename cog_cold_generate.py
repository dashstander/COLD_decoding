from cog import BasePredictor, Input, Path
from dataclasses import dataclass
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from bleuloss import batch_log_bleulosscnn_ae
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import Tuple

from util import (
    get_text_from_logits,
    initialize,
    rank_and_filter,
    one_hot, post_process,
    post_sent,
    decode_with_model_topk,
    soft_forward,
    soft_nll,
    top_k_filter_3d
)
from modeling_opengpt2 import OpenGPT2LMHeadModel


stop_words = set(stopwords.words('english'))


@dataclass
class CounterfactualConfig:
    init_mode: str = 'random'
    init_temp: float = 1
    length: int = 1
    max_length: int = 50
    num_iters: int = 2000
    min_iters: int = 10
    constraint_weight: float = 0.2
    counterfactual_max_ngram: int =  3
    stepsize: float = 0.1
    noise_iters: int = 1
    win_anneal_iters: int = 1000
    start: int = 0
    end: int = 5
    lr_nll_portion: float = 0.9
    topk: int = 5
    output_lgt_temp: int = 1
    large_noise_iters: Tuple[int] = (50, 200, 500)
    large_gs_std: Tuple[int] = (0.5,0.1,0.05)
    stepsize_ratio: int = 1


def counterfactual_decode(model, tokenizer, device, x="", z="", config=None, model_back=None):
    '''
    x: left context   (prompt in lexical lexical task)
    z: optimization target  (original ending in counterfactual task)
    constraints: (constraint set in lexical constrained task)
    '''
    x_ = tokenizer.encode(x)
    x_t = torch.tensor(x_, device=device, dtype=torch.long)
    x_onehot = one_hot(x_t, dimension=tokenizer.vocab_size)
    # repeat batch_size times
    x_t = x_t.unsqueeze(0).repeat(1, 1)
    x_onehot = x_onehot.repeat(1, 1, 1)
    z_mask = None    
    z_ = tokenizer.encode(z)[1:]  # delete the "." token we appended before
    z_t = torch.tensor(z_, device=device, dtype=torch.long)
    z_onehot = one_hot(z_t, dimension=tokenizer.vocab_size)
    z_onehot = z_onehot.repeat(1, 1, 1)

    z_t = z_t.unsqueeze(0).repeat(1, 1)

    length = config.length
    if length <= 0:
        length = z_t.shape[1] - length

    z_words = word_tokenize(z[2:])  # delete the ". " token we appended before
    z_nonstop_words = [w.lower() for w in z_words if w.lower() not in stop_words and w.isalnum()]
    z_nonstop_words += [z_words[0]]  # add the first token
    z_nonstop_words = ' ' + ' '.join(z_nonstop_words)
    z_nonstop_ = tokenizer.encode(z_nonstop_words)

    z_mask = np.zeros([tokenizer.vocab_size])
    z_mask[z_nonstop_] = 1.
    z_mask = torch.tensor(z_mask, device=device)
    z_mask = z_mask.unsqueeze(0).unsqueeze(0).repeat(1, length, 1)

    if config.init_mode == 'random':
        init_logits = initialize(model, x_t, length, config.init_temp, device)
    else:
        init_logits = z_onehot / 0.1
        init_logits = init_logits[:, :length, :]
        if length > init_logits.shape[1]:
            init_logits = torch.cat(
                [init_logits,
                 torch.zeros([1, length - init_logits.shape[1], tokenizer.vocab_size], device=device)],
                dim=1)
    text, _, _ = get_text_from_logits(init_logits, tokenizer)
   
    
    y_logits = init_logits
    epsilon = torch.nn.Parameter(torch.zeros_like(y_logits))
    optim = torch.optim.Adam([epsilon], lr=config.stepsize)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optim,
        step_size=config.stepsize_iters,
        gamma=config.stepsize_ratio
    )
    frozen_len = config.frozen_length

    y_logits_ = None
    noise_std = 0.0

    ## Encode x beforehand
    soft_forward_x = x_onehot[:, -1:, :]  # The last token of x is used in soft_forward
    if x_t.shape[1] == 1:
        x_model_past = None
    else:
        x_model_outputs = model(x_t[:, :-1])
        x_model_past = x_model_outputs.past_key_values
        x_model_past = [_.detach() for _ in x_model_past]

    # For right to left model
    rl_reverse_index = torch.arange(y_logits.shape[1] - 1, -1, -1)

    mask_t = None

    for iter in range(config.num_iters):
        optim.zero_grad()
        y_logits_ = y_logits + epsilon

        soft_forward_y = y_logits_ / 0.001
        if config.straight_through:
            if mask_t is None:
                soft_forward_y = (y_logits_.detach() / 0.001 - y_logits_).detach() + y_logits_
            else:
                soft_forward_y = top_k_filter_3d(y_logits_, config.topk, mask=mask_t, extra_mask=z_mask) / 0.001

        y_logits_t = soft_forward(model, soft_forward_x, soft_forward_y, x_past=x_model_past)

        if config.topk == 0:
            mask_t = None
        else:
            _, indices_t = torch.topk(y_logits_t, config.topk)
            mask_t = torch.zeros_like(y_logits_t).scatter_(2, indices_t, 1)

        # Compute loss, gradients, and update.
        lr_nll_loss = soft_nll(
            top_k_filter_3d(y_logits_t / config.output_lgt_temp, config.topk, extra_mask=z_mask),
            y_logits_ / config.input_lgt_temp)

        if config.lr_nll_portion == 1.0:
            rl_nll_loss = lr_nll_loss
        else:
            # add right-to-left model (rl)
            y_logits_rev = y_logits_[:, rl_reverse_index, :]
            y_logits_rev_t = model_back(y_logits_rev.argmax(-1) + 1).logits[:, :-1, :]
            y_logits_rev_t = y_logits_rev_t[:, :, 1:y_logits_.shape[-1] + 1]
            rl_nll_loss = soft_nll(
                top_k_filter_3d(y_logits_rev_t / config.output_lgt_temp, config.rl_topk),
                y_logits_rev[:, 1:] / config.input_lgt_temp)
        c_loss = batch_log_bleulosscnn_ae(
            decoder_outputs=top_k_filter_3d(y_logits_, config.topk, mask=mask_t, extra_mask=z_mask).transpose(0, 1),
            target_idx=z_t,
            ngram_list=list(range(2, config.counterfactual_max_ngram + 1))
        )

        loss = (1.0 - config.constraint_weight) * config.lr_nll_portion * lr_nll_loss \
               + (1.0 - config.constraint_weight) * (1 - config.lr_nll_portion) * rl_nll_loss \
               + config.constraint_weight * c_loss
        loss = loss.mean()

        if iter < config.num_iters - 1:  # so that the mask_t at the last iteration will not change
            loss.backward()
            optim.step()
            scheduler.step()  # turn off the scheduler
            last_lr = scheduler.get_last_lr()[0]

        ## noise
        if iter < config.num_iters - 1:
            large_noise_iters =  list(config.large_noise_iters)
            large_gs_stds = list(config.large_gs_std)
            noise_std = 0.
            if iter % config.noise_iters == 0:
                noise_last = True
                for ni in range(len(large_noise_iters)):
                    if iter < large_noise_iters[ni]:
                        noise_last = False
                        break
                if noise_last:
                    noise_std = config.gs_std
                else:
                    noise_std = large_gs_stds[ni]

                noise = torch.normal(
                    mean=config.gs_mean,
                    std=noise_std,
                    size=epsilon.size(),
                    device='cuda',
                    requires_grad=False
                )
                if config.win_anneal_iters >= 0 and iter >= config.win_anneal_iters:
                    zeros = torch.zeros_like(noise)
                    noise_mix = torch.cat([zeros[:, :frozen_len], noise[:, frozen_len:]], dim=1)
                    y_logits = y_logits + noise_mix
                else:
                    y_logits = y_logits + noise

    text, _, last_text_ids = decode_with_model_topk(
        model, y_logits_, config.topk, soft_forward_x, x_model_past, tokenizer, extra_mask=z_mask)

    last_rank_loss = model(input_ids=last_text_ids, labels=last_text_ids).loss
    last_rank_loss = last_rank_loss.detach().clone().data.cpu().numpy()
    text_post = post_process(last_text_ids, model, config.max_length, config.length, tokenizer, device)
    ppl_last = np.exp(last_rank_loss)

    return ppl_last, text, text_post




class ColdDecoder(BasePredictor):

    def setup(self):
        pretrained_model = 'gpt2-large'
        back_model = 'danyaljj/opengpt2_pytorch_backward'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.forward_model = GPT2LMHeadModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True,
            resid_pdrop=0,
            embd_pdrop=0,
            attn_pdrop=0,
            summary_first_dropout=0
        )
        self.model.to(self.device)
        self.model.eval()
        # Freeze GPT-2 weights
        for param in self.forward_model.parameters():
            param.requires_grad = False

        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

        self.backward_model = OpenGPT2LMHeadModel.from_pretrained(
            back_model,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
            summary_first_dropout=0
        )
        self.backward_model.to(self.device)
        self.backward_model.eval()
        # Freeze GPT-2 weights
        for param in self.backward_model.parameters():
            param.requires_grad = False

    def predict(
        self,
        input: Path = Input(description="Grayscale input image"),
        scale: float = Input(
            description="Factor to scale image by", gt=0, lt=10, default=1.5
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        # processed_input = preprocess(input)
        # output = self.model(processed_input, scale)
        # return postprocess(output)
        pass
    

    def counterfactual_reasoning(self, premise, counterfactual, orig_ending):
        x = premise + ' ' + counterfactual
        ori_endings = self.tokenize.sent_tokenize(orig_ending)
        x_text_so_far = [""]
        x_addon = [[x]]
        outputs = []
        for oi, z_sent in enumerate(ori_endings):
            z_text_so_far = z_sent.strip()
            z_text_so_far = ". " + z_text_so_far

            assert len(x_text_so_far) == len(x_addon), "%d vs %d" % (len(x_text_so_far), len(x_addon))

            new_x_text_so_far = []
            new_x_addon = []
            for ii, text_i in enumerate(x_text_so_far):
                for text_j in x_addon[ii]:
                    text_ij = text_i.strip() + " " + text_j.strip()
                    new_x_text_so_far.append(text_ij)

                    text_ij = text_ij.strip()

                    _, _, text_post = counterfactual_decode(
                        self.model,
                        self.tokenizer,
                        self.device,
                        text_ij,
                        z_text_so_far,
                        config,
                        model_back=self.backward_model
                    )

                    outputs.append([text_ij, text_post])

                    #  Rank and filter text_post from util.py:
                    text_post = [post_sent(x) for x in text_post]
                    text_post = rank_and_filter(
                        text_post,
                        text_ij,
                        z_text_so_far,
                        self.model,
                        self.tokenizer,
                        self.device,
                        config.no_loss_rerank
                    )

                    if ii == len(x_text_so_far) - 1 and oi == len(ori_endings) - 1:
                        last_output = text_post
                        final_res = ' '.join([text_ij, last_output])
                        outputs.append(final_res)
                       
                    new_x_addon.append([text_post])

            x_text_so_far = new_x_text_so_far
            x_addon = new_x_addon

            break

        complete_output = outputs
        out = {
            'premise': premise,
            'initial': d.get('initial', ""),
            'counterfactual': counterfactual,
            'original_ending': orig_ending,
            'generation_complete': complete_output,
        }
        return out

     
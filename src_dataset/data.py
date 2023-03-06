
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import torch
import logging
import pandas as pd
import numpy as np
import random
import json

from torch.utils.data import Dataset
from collections import Counter
from typing import List, Tuple, Dict, Set
from collections import defaultdict as ddict

from transformers import AutoTokenizer, AutoModel

logging.basicConfig(level=logging.INFO)

class DatasetCanonicalizationStatic(Dataset):
    def __init__(self, data, w2i):
        self.data = data
        self.w2i = w2i

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        head, rel, tail = self.data[item]['h']['name'], self.data[item]['r']['name'] , self.data[item]['t']['name']    # (h,r,t)
        # head_c, rel_c, tail_c = self.data[item]['cluster']   # (h,r,t)
        h = np.array([self.w2i[w] for w in head])
        r = np.array([self.w2i[w] for w in rel])
        t = np.array([self.w2i[w] for w in tail])
        id = self.data[item]['tri_id']

        return h, r, t, id


class DatasetCanonicalizationContextual(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.max_len = args.max_len
        self.args = args
        self.model_name = args.plm_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def __len__(self):
        return len(self.data)

    def triple2ids(self, triple):

        head, rel, tail = triple
        head_toks = self.tokenizer.tokenize(head)
        rel_toks = self.tokenizer.tokenize(rel)
        tail_toks = self.tokenizer.tokenize(tail)
        if self.args.sep_strategy == 'sep':
            # if self.args.plm_model == 'gpt'
            all_tok = [self.tokenizer.cls_token] + head_toks + [self.tokenizer.sep_token] + rel_toks + [self.tokenizer.sep_token] + tail_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            assert len(all_tok) <= self.max_len

            attention_mask = len(all_tok) * [1] +  (self.max_len - len(all_tok)) * [0]
            scatter_mask = [0]+[1]*len(head_toks)+[0]+[2]*len(rel_toks)+[0]+[3]*len(tail_toks)+[0] + (self.max_len - len(all_tok)) * [0]
            all_tok = all_tok + (self.max_len - len(all_tok)) * [self.tokenizer.pad_token]
            all_ids = self.tokenizer.convert_tokens_to_ids(all_tok)
            h_r_t_position = [1, len(head_toks)+2, len(head_toks)+1+len(rel_toks)+2]
        elif self.args.sep_strategy == 'none':
            if 'gpt' in self.model_name:
                all_tok = head_toks + rel_toks + tail_toks + [self.tokenizer.eos_token] # use cls as first tok or sep?
                assert len(all_tok) <= self.max_len
                attention_mask = len(all_tok) * [1] +  (self.max_len - len(all_tok)) * [0]
                scatter_mask = [1]*len(head_toks)+[2]*len(rel_toks)+[3]*len(tail_toks)+[0]+ (self.max_len - len(all_tok)) * [0]
                all_tok = all_tok + (self.max_len - len(all_tok)) * [self.tokenizer.eos_token] # will mask it out
                all_ids = self.tokenizer.convert_tokens_to_ids(all_tok)
                h_r_t_position = [0, len(head_toks), len(head_toks)+len(rel_toks)]
            else:
                all_tok = [self.tokenizer.cls_token] + head_toks + rel_toks + tail_toks + [
                    self.tokenizer.sep_token]  # use cls as first tok or sep?
                assert len(all_tok) <= self.max_len

                attention_mask = len(all_tok) * [1] + (self.max_len - len(all_tok)) * [0]
                scatter_mask = [0] + [1] * len(head_toks) + [2] * len(rel_toks) + [3] * len(tail_toks) + [0] + (
                            self.max_len - len(all_tok)) * [0]
                all_tok = all_tok + (self.max_len - len(all_tok)) * [self.tokenizer.pad_token]
                all_ids = self.tokenizer.convert_tokens_to_ids(all_tok)
                h_r_t_position = [1, len(head_toks) + 1, len(head_toks) + 1 + len(rel_toks)]

        elif self.args.sep_strategy == 'prompt-none':
            all_tok = [self.tokenizer.cls_token] + ['the', self.tokenizer.mask_token] + head_toks + rel_toks + ['the', self.tokenizer.mask_token] + tail_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            assert len(all_tok) <= self.max_len

            attention_mask = len(all_tok) * [1] +  (self.max_len - len(all_tok)) * [0]
            scatter_mask = [0]*2+[1]+[0]*len(head_toks)+[2]*len(rel_toks)+[0]+[3]+[0]*len(tail_toks)+[0]+ (self.max_len - len(all_tok)) * [0]
            all_tok = all_tok + (self.max_len - len(all_tok)) * [self.tokenizer.pad_token]
            all_ids = self.tokenizer.convert_tokens_to_ids(all_tok)
            h_r_t_position = [3, len(head_toks)+3, len(head_toks)+4+len(rel_toks)]
        elif self.args.sep_strategy == 'split':
            h_r_t_position = [1, 1, 1]

            all_tok_h = [self.tokenizer.cls_token] + head_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            attention_mask_h = len(all_tok_h) * [1] +  (self.max_len - len(all_tok_h)) * [0]
            scatter_mask_h = [0]+[1]*len(head_toks)+[0]+ (self.max_len - len(all_tok_h)) * [0]
            all_tok_h = all_tok_h + (self.max_len - len(all_tok_h)) * [self.tokenizer.pad_token]
            all_ids_h = self.tokenizer.convert_tokens_to_ids(all_tok_h)

            all_tok_r = [self.tokenizer.cls_token] + rel_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            attention_mask_r = len(all_tok_r) * [1] + (self.max_len - len(all_tok_r)) * [0]
            scatter_mask_r = [0] + [1] * len(rel_toks) + [0] + (self.max_len - len(all_tok_r)) * [0]
            all_tok_r = all_tok_r + (self.max_len - len(all_tok_r)) * [self.tokenizer.pad_token]
            all_ids_r = self.tokenizer.convert_tokens_to_ids(all_tok_r)

            all_tok_t = [self.tokenizer.cls_token] + tail_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            attention_mask_t = len(all_tok_t) * [1] + (self.max_len - len(all_tok_t)) * [0]
            scatter_mask_t = [0] + [1] * len(tail_toks) + [0] + (self.max_len - len(all_tok_t)) * [0]
            all_tok_t = all_tok_t + (self.max_len - len(all_tok_t)) * [self.tokenizer.pad_token]
            all_ids_t = self.tokenizer.convert_tokens_to_ids(all_tok_t)

            all_ids = [all_ids_h, all_ids_r, all_ids_t]
            attention_mask = [attention_mask_h, attention_mask_r, attention_mask_t]
            scatter_mask = [scatter_mask_h, scatter_mask_r, scatter_mask_t]
        elif self.args.sep_strategy == 'prompt-split':
            h_r_t_position = [2, 1, 2] # mask pos, start rel pos, mask pos

            all_tok_h = [self.tokenizer.cls_token] + ['the'] + [self.tokenizer.mask_token] + head_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            attention_mask_h = len(all_tok_h) * [1] +  (self.max_len - len(all_tok_h)) * [0]
            scatter_mask_h = [0]*2+[1]+[0]*len(head_toks)+[0]+ (self.max_len - len(all_tok_h)) * [0]
            all_tok_h = all_tok_h + (self.max_len - len(all_tok_h)) * [self.tokenizer.pad_token]
            all_ids_h = self.tokenizer.convert_tokens_to_ids(all_tok_h)

            all_tok_r = [self.tokenizer.cls_token] + rel_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            attention_mask_r = len(all_tok_r) * [1] + (self.max_len - len(all_tok_r)) * [0]
            scatter_mask_r = [0] + [1] * len(rel_toks) + [0] + (self.max_len - len(all_tok_r)) * [0]
            all_tok_r = all_tok_r + (self.max_len - len(all_tok_r)) * [self.tokenizer.pad_token]
            all_ids_r = self.tokenizer.convert_tokens_to_ids(all_tok_r)
            all_tok_t = [self.tokenizer.cls_token] + ['the'] + [self.tokenizer.mask_token] + tail_toks + [self.tokenizer.sep_token] # use cls as first tok or sep?
            attention_mask_t = len(all_tok_t) * [1] + (self.max_len - len(all_tok_t)) * [0]
            scatter_mask_t = [0]*2+[1]+[0]* len(tail_toks) + [0] + (self.max_len - len(all_tok_t)) * [0]
            all_tok_t = all_tok_t + (self.max_len - len(all_tok_t)) * [self.tokenizer.pad_token]
            all_ids_t = self.tokenizer.convert_tokens_to_ids(all_tok_t)
            all_ids = [all_ids_h, all_ids_r, all_ids_t]
            attention_mask = [attention_mask_h, attention_mask_r, attention_mask_t]
            scatter_mask = [scatter_mask_h, scatter_mask_r, scatter_mask_t]


        return all_ids, attention_mask, scatter_mask, h_r_t_position



    def triple2ids_sent(self, tokens, h_pos, r_pos, t_pos):
        head = ' '.join(tokens[h_pos[0]: h_pos[1]])
        rel = ' '.join(tokens[r_pos[0]: r_pos[1]])
        tail = ' '.join(tokens[t_pos[0]: t_pos[1]])

        if not ((h_pos[1] <= r_pos[0]) and (r_pos[1] <= t_pos[0])):
            logging.info('error')

        left_span = ' '.join(tokens[:h_pos[0]])
        mid_span_hr = ' '.join(tokens[h_pos[1]:r_pos[0]])
        mid_span_rt = ' '.join(tokens[r_pos[1]:t_pos[0]])
        right_span = ' '.join(tokens[t_pos[1]:])
        head_toks = self.tokenizer.tokenize(head)
        rel_toks = self.tokenizer.tokenize(rel)
        tail_toks = self.tokenizer.tokenize(tail)
        left_span_toks = self.tokenizer.tokenize(left_span)
        mid_span_hr_toks = self.tokenizer.tokenize(mid_span_hr)
        mid_span_rt_toks = self.tokenizer.tokenize(mid_span_rt)
        right_span_toks = self.tokenizer.tokenize(right_span)

        if 'gpt' in self.model_name:
            all_tok =  left_span_toks + head_toks + mid_span_hr_toks + rel_toks + mid_span_rt_toks + tail_toks + right_span_toks + [
                          self.tokenizer.eos_token]
            attention_mask = len(all_tok) * [1] + (self.max_len - len(all_tok)) * [0]
            scatter_mask = [0] * (len(left_span_toks)) + [1] * len(head_toks) + [0] * len(mid_span_hr_toks) + [
                2] * len(rel_toks) + [0] * len(mid_span_rt_toks) + [3] * len(tail_toks) + [0] * (
                                       1 + len(right_span_toks)) + (self.max_len - len(all_tok)) * [0]
            all_tok = all_tok + (self.max_len - len(all_tok)) * [self.tokenizer.eos_token] # will mask it out
            all_ids = self.tokenizer.convert_tokens_to_ids(all_tok)
            h_r_t_position = [len(left_span_toks), len(left_span_toks) + len(head_toks) + len(mid_span_hr_toks),
                              len(left_span_toks) + len(head_toks) + len(mid_span_hr_toks) + len(rel_toks) + len(mid_span_rt_toks)]
        else:
            all_tok = [self.tokenizer.cls_token] + left_span_toks + head_toks + mid_span_hr_toks + rel_toks + mid_span_rt_toks + tail_toks + right_span_toks + [self.tokenizer.sep_token]

            attention_mask = len(all_tok) * [1] + (self.max_len - len(all_tok)) * [0]
            scatter_mask = [0]*(1+len(left_span_toks))+[1]*len(head_toks)+[0]*len(mid_span_hr_toks)+[2]*len(rel_toks)+[0]*len(mid_span_rt_toks)+[3]*len(tail_toks)+[0]*(1+len(right_span_toks))+ (self.max_len - len(all_tok)) * [0]
            all_tok = all_tok + (self.max_len - len(all_tok)) * [self.tokenizer.pad_token]
            all_ids = self.tokenizer.convert_tokens_to_ids(all_tok)
            h_r_t_position = [1 + len(left_span_toks), 1 + len(left_span_toks) + len(head_toks) + len(mid_span_hr_toks), 1 + len(left_span_toks) + len(head_toks) + len(mid_span_hr_toks) + len(rel_toks) + len(mid_span_rt_toks)]
            if len(all_tok) > self.max_len:
                return self.triple2ids((head, rel, tail))

        return all_ids, attention_mask, scatter_mask, h_r_t_position



    def __getitem__(self, item):
        if self.args.sep_strategy in ['sep', 'none', 'split', 'prompt-split', 'prompt-none']:

            head, rel, tail = self.data[item]['h']['name'], self.data[item]['r']['name'] , self.data[item]['t']['name']
            head, rel, tail = ' '.join(head), ' '.join(rel), ' '.join(tail)
            all_ids, attention_mask, scatter_mask, h_r_t_position = self.triple2ids((head, rel, tail))
            id = self.data[item]['tri_id']

        else: # sentence
            tokens = self.data[item]['text']
            h_pos, r_pos, t_pos = self.data[item]['h']['pos'], self.data[item]['r']['pos'], self.data[item]['t']['pos']
            all_ids, attention_mask, scatter_mask, h_r_t_position = self.triple2ids_sent(tokens, h_pos, r_pos, t_pos)
            id = self.data[item]['tri_id']


        return np.array(all_ids), np.array(attention_mask), np.array(scatter_mask), np.array(h_r_t_position), id

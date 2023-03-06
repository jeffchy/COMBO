#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging

from torch import nn
from collections import Counter
import datetime, time, pickle
import numpy as np
import os
import json
from tqdm import tqdm


def read_file(f_name: str):
    if f_name is None or not os.path.exists(f_name): return FileNotFoundError
    with open(f_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def merge_dict(a, b):
    for k, v in b.items():
        a[k] = v
    return a

def get_static_vocabs(data):
    logging.info("get static embedding")
    vocab = Counter()
    triple = ['h', 'r', 't']
    for i in data:
        for t in triple:
            toks = i[t]['name']
            for t in toks:
                vocab.update([t])

    vocab = ['<PAD>'] + ['<UNK>'] + list(vocab.keys())
    i2w = {i: w for i, w in enumerate(vocab)}
    w2i = {w: i for i, w in enumerate(vocab)}

    return vocab, i2w, w2i


class _CustomDataParallel(nn.Module):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__()
        self.model = nn.DataParallel(model).cuda()
        print(type(self.model))

    def forward(self, *input, **kwargs):
        outputs = self.model(*input, **kwargs)

        return outputs

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

def create_datetime_str():
    datetime_dt = datetime.datetime.today()
    datetime_str = datetime_dt.strftime("%m%d%H%M%S")
    datetime_str = datetime_str + '_' + str(time.time())
    return datetime_str

def save_pkl(obj, path):
    logging.info("saving at: {}".format(path))
    pickle.dump(obj, open(path, 'wb'))

def load_pkl(path):
    logging.info("loading at: {}".format(path))
    return pickle.load(open(path, 'rb'))

def make_glove_embed(dataset_path, i2t, glove_path, embed_dim=100):

    path = os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim))
    logging.info("try to load embed at path: {}".format(path))
    if os.path.exists(path):
        logging.info("path exists, load embed and return")
        return load_glove_embed(dataset_path, embed_dim)

    glove = {}
    vecs = [] # use to produce unk

    # load glove
    with open(os.path.join(glove_path, 'glove.6B.{}d.txt'.format(embed_dim)),
              'r', encoding='utf-8') as f:
        for line in tqdm(f.readlines()):
            split_line = line.split()
            word = split_line[0]
            embed_str = split_line[1:]
            try:
                # ignore error
                embed_float = [float(i) for i in embed_str]
            except:
                continue

            if word not in glove:
                glove[word] = embed_float
                vecs.append(embed_float)

    embed = []
    for i in tqdm(i2t):
        word = i2t[i].lower()
        if word in glove:
            embed.append(glove[word])
        else:
            embed.append(np.random.normal(size=(embed_dim)))

    final_embed = np.array(embed, dtype=np.float)
    num_error = np.sum(final_embed[final_embed == 0])
    logging.info("NUM ERROR: {}".format(num_error))
    pickle.dump(final_embed, open(os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim)), 'wb'))
    logging.info("SAVED")
    return final_embed


def load_glove_embed(dataset_path, embed_dim):
    """
    :param config:
    :return the numpy array of embedding of task vocabulary: V x D:
    """
    return pickle.load(open(os.path.join(dataset_path, 'glove.{}.emb'.format(embed_dim)), 'rb'))


def create_vocab_and_save(data, dataset_dir):
    vocab, i2w, w2i = get_static_vocabs(data)
    save_pkl(vocab, os.path.join(dataset_dir, 'vocab.pkl'))
    save_pkl(i2w, os.path.join(dataset_dir, 'i2w.pkl'))
    save_pkl(w2i, os.path.join(dataset_dir, 'w2i.pkl'))


def invertDict(curr_map):
    if not curr_map: return dict()
    entry = list(curr_map.items())[0]
    if type(entry[0]) in [str, int] and type(entry[1]) in [list, set]:
        inv_map = dict()
        for key, value_collection in curr_map.items():
            for value in value_collection:
                inv_map[value] = inv_map.get(value, set())
                inv_map[value].add(key)
    elif all(map(lambda z: type(z) is str, entry)):
        inv_map = dict()
        for key, value in curr_map.items():
            inv_map[value] = inv_map.get(value, set())
            inv_map[value].add(key)
    else:
        raise NotImplementedError('{0}:{1}'.format(type(entry[0]), type(entry[1])))
    return inv_map

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


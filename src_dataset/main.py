import argparse
import logging
import sys, os

import pandas as pd

sys.path.append('../')
from src_dataset.data import DatasetCanonicalizationStatic, DatasetCanonicalizationContextual
from src_dataset.utils import save_pkl, load_pkl, merge_dict, make_glove_embed, load_glove_embed, mkdir, create_datetime_str, read_file, get_static_vocabs, create_vocab_and_save
from torch.utils.data import DataLoader, Dataset
from src_dataset.static_embed import StaticEmbedder
from src_dataset.contextual_embed import ContextualEmbedder
from tqdm import tqdm
from src_dataset.cluster import Cluster
import numpy as np


class Experiment:

    def __init__(self, args):
        self.args = args
        self.test_data = read_file(os.path.join(self.args.dataset_dir, self.args.test_file))
        self.valid_data = read_file(os.path.join(self.args.dataset_dir, self.args.valid_file))

        # create_vocab_and_save(self.test_data+self.valid_data, self.args.dataset_dir)
        self.i2w = load_pkl(os.path.join(self.args.dataset_dir, 'i2w.pkl'))
        self.w2i = load_pkl(os.path.join(self.args.dataset_dir, 'w2i.pkl'))
        self.cluster = None
        self.save_dir = os.path.join(self.args.dataset_dir, self.args.save_dir)
        self.plm_layers = [int(i) for i in self.args.plm_layer.split(',')]
        if self.args.embedding != 'contextual': self.plm_layers = [0]
        mkdir(self.save_dir)

    @staticmethod
    def get_id2cluster(data):
        id2cluster = {}
        for i in data:
            id = i['tri_id']
            h_c, r_c, t_c = i['h']['id'], i['r']['label'], i['t']['id']
            h_ci, t_ci = i['h']['instance'], i['t']['instance']

            id2cluster['{}{}'.format(id, 'h')] = h_c
            id2cluster['{}{}'.format(id, 'r')] = r_c
            id2cluster['{}{}'.format(id, 't')] = t_c
            id2cluster['{}{}'.format(id, 'hi')] = h_ci
            id2cluster['{}{}'.format(id, 'ti')] = t_ci

        return id2cluster


    def get_static_emb(self, dset='valid'):
        assert dset in ['valid', 'test']
        if dset == 'valid':
            data = self.valid_data
        else:
            data = self.test_data

        self.dataset = DatasetCanonicalizationStatic(data, self.w2i)
        # 1 for static embedding
        self.dataloader = DataLoader(self.dataset, 1)

        emb = make_glove_embed(self.args.dataset_dir, self.i2w, self.args.glove_path, self.args.embed_dim)

        self.encoder = StaticEmbedder(emb, self.args)
        id2cluster = self.get_id2cluster(data)
        id2emb = {0: {}}

        for b in tqdm(self.dataloader, desc="encoding {} set".format(dset)):
            h, r, t, id = b
            h_e = self.encoder(h)
            r_e = self.encoder(r)
            t_e = self.encoder(t)
            id2emb[0]['{}{}'.format(id.item(), 'h')] = h_e.cpu().detach().numpy()
            id2emb[0]['{}{}'.format(id.item(), 'r')] = r_e.cpu().detach().numpy()
            id2emb[0]['{}{}'.format(id.item(), 't')] = t_e.cpu().detach().numpy()
            id2emb[0]['{}{}'.format(id.item(), 'hi')] = h_e.cpu().detach().numpy()
            id2emb[0]['{}{}'.format(id.item(), 'ti')] = t_e.cpu().detach().numpy()

        return id2cluster, id2emb

    def get_contextual_emb(self, dset='valid'):
        assert dset in ['valid', 'test']
        if dset == 'valid':
            data = self.valid_data
        else:
            data = self.test_data

        self.dataset = DatasetCanonicalizationContextual(data, args=self.args)
        # 1 for static embedding
        self.dataloader = DataLoader(self.dataset, self.args.encode_bz)

        self.encoder = ContextualEmbedder(self.args).cuda()
        id2cluster = self.get_id2cluster(data)
        id2emb = {}

        for b in tqdm(self.dataloader, desc="encoding {} set".format(dset)):
            all_ids, attention_mask, scatter_mask, h_r_t_position, id = b
            all_ids = all_ids.cuda()
            attention_mask = attention_mask.cuda()
            scatter_mask = scatter_mask.cuda()
            embeds = self.encoder(all_ids, attention_mask, scatter_mask)
            B = id.size()[0]
            id = id.numpy()
            for layer, (head_batch_emb, rel_batch_emb, tail_batch_emb) in embeds.items():
                head_batch_emb = head_batch_emb.cpu().detach().numpy()
                rel_batch_emb = rel_batch_emb.cpu().detach().numpy()
                tail_batch_emb = tail_batch_emb.cpu().detach().numpy()
                if layer not in id2emb:
                    id2emb[layer] = {}
                for i in range(B):
                    id2emb[layer]['{}{}'.format(id[i], 'h')] = head_batch_emb[i]
                    id2emb[layer]['{}{}'.format(id[i], 'r')] = rel_batch_emb[i]
                    id2emb[layer]['{}{}'.format(id[i], 't')] = tail_batch_emb[i]
                    id2emb[layer]['{}{}'.format(id[i], 'hi')] = head_batch_emb[i]
                    id2emb[layer]['{}{}'.format(id[i], 'ti')] = tail_batch_emb[i]

        return id2cluster, id2emb

    def find_best_val(self, metrics, keys=('macro_f1', 'micro_f1', 'pair_f1')):
        best_val = 0
        best_metric = None
        best_th = None
        for m in metrics:
            s = 0
            for k in keys:
                s += m[k]
            s /= len(keys)
            if s > best_val:
                best_val = s
                best_th = m['th']
                best_metric = m

        return best_th, best_metric

    def run(self, modes):
        modes = modes.split(',')
        res = {'args': self.args}
        # validation
        time_str = create_datetime_str()
        get_emb_func = self.get_static_emb if self.args.embedding in ['static', 'random'] else self.get_contextual_emb

        id2cluster_val, id2emb_val = get_emb_func(dset='valid')
        self.cluster_val = Cluster(self.args, id2emb=id2emb_val, id2cluster=id2cluster_val, modes=modes)

        id2cluster_test, id2emb_test = get_emb_func(dset='test')
        self.cluster_test = Cluster(self.args, id2emb=id2emb_test, id2cluster=id2cluster_test, modes=modes)

        for layer in self.plm_layers:
            if layer not in res:
                res[layer] = {}

            for mode in modes:

                logging.info("======== clustering mode {} layer {} =========".format(mode, layer))
                # validate round 1
                metrics = []
                for th in np.arange(0.1, 1.5, 0.1):
                    metric = self.cluster_val.cluster(mode, layer, th)
                    metric['th'] = th
                    metrics.append(metric)
                    if metric['n_pred'] < metric['n_gold'] * 0.7:
                        print('too few pred clusters ... ')
                        break

                logging.info("round I matrics: \n{}".format(metrics))
                best_th, best_metric = self.find_best_val(metrics, keys=('macro_f1', 'micro_f1', 'pair_f1'))
                logging.info("round I best metrics: {}, th: {}".format(best_metric, best_th))

                # validate round 2
                for th in np.arange(best_th-0.1, best_th+0.1, 0.03):
                    metric = self.cluster_val.cluster(mode, layer, th)
                    metric['th'] = th
                    metrics.append(metric)

                logging.info("round II matrics: \n{}".format(metrics))

                best_th, best_metric = self.find_best_val(metrics, keys=('macro_f1', 'micro_f1', 'pair_f1'))
                res[layer]['val_metrics_{}'.format(mode)] = metrics
                res[layer]['best_val_metrics_{}'.format(mode)] = best_metric
                res[layer]['best_val_th_{}'.format(mode)] = best_th

                logging.info("round II best metrics: {}, th: {}".format(best_metric, best_th))
                # test
                metric = self.cluster_test.cluster(mode, layer, best_th)

                if mode in ['hi', 'ti', 'hiti']:
                    metrics1 = self.cluster_test.cluster_and_evaluate_hierarchy(mode, layer)
                    metric = merge_dict(metric, metrics1)
                    # p = os.path.join(self.save_dir,
                    #          '{}.{}.{}'.format(create_datetime_str(), mode,'cluster.pkl'))
                    # print("cluster path: {}".format(p))
                    # save_pkl(self.cluster_test.clustering, p)

                logging.info("test metrics for layer {}: {}".format(layer, metric))
                res[layer]['test_metrics_{}'.format(mode)] = metric


        path = os.path.join(self.save_dir, time_str+'.res.pkl')
        save_pkl(res, path)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='../combo/')
    parser.add_argument('--test_file', type=str, default='revised.test.json')
    parser.add_argument('--valid_file', type=str, default='revised.val.json')
    parser.add_argument('--embedding', type=str, default='static')
    parser.add_argument('--save_dir', type=str, default='save')

    ##### Setting
    parser.add_argument('--mode', type=str, default='h,r,t')

    ##### Static
    parser.add_argument('--glove_path', type=str, default='../glove/')
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--encode_bz', type=int, default=32)

    ##### Contextual
    parser.add_argument('--plm_model', type=str, default='bert-base-uncased')
    parser.add_argument('--plm_layer', type=str, default='12')
    parser.add_argument('--sep_strategy', type=str, default='sep', help="[cls] h r t (none)/ [cls] h [sep] r [sep] t (sep)")
    parser.add_argument('--max_len', type=int, default=128, help="max_len")
    parser.add_argument('--rep_strategy', type=str, default='mean', help="span rep strategy, mean, max, diffsum, cat")


    ##### Clustering
    parser.add_argument('--pca', type=int, default=350, help="PCA components")
    parser.add_argument('--standardize', type=int, default=1, help="If we use standization")
    parser.add_argument('--r_linkage', type=str, default='complete', help="If we use complete or single linkage")

    args = parser.parse_args()
    assert args.rep_strategy in ['mean', 'max', 'diffsum', 'cat']
    assert args.embedding in ['static', 'contextual', 'random']
    assert args.sep_strategy in ['sep', 'none', 'sentence', 'split', 'prompt-split', 'prompt-none', 'prompt-sent']

    if 'gpt' in args.plm_model:
        assert args.sep_strategy in ['sentence', 'none']

    logging.info("running in mode: {}".format(args.mode))
    if args.sep_strategy == 'sentence': args.max_len = 128
    exp = Experiment(args)


    exp.run(modes=args.mode)





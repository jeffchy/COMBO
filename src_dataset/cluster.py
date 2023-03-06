from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import logging
import numpy as np
from src_dataset.metric import get_metrics, get_instance_metrics
from src_dataset.utils import invertDict
from tqdm import tqdm

class Cluster:
    def __init__(self, args, id2emb, id2cluster, modes):
        self.args = args
        self.id2emb = id2emb
        self.id2cluster = id2cluster
        self.mode_emb_cluster = {}
        self.plm_layers = [int(i) for i in self.args.plm_layer.split(',')]
        if self.args.embedding != 'contextual': self.plm_layers = [0]

        for layer in self.plm_layers:
            if layer not in self.mode_emb_cluster:
                self.mode_emb_cluster[layer] = {}
                for mode in modes:
                    assert mode in ['ht', 'h', 't', 'r', 'hrt', 'hi','ti', 'hiti']
                    emb, cluster = self.get_emb_cluster(mode, layer)
                    idxs = [k for k, v in emb.items()]
                    emb_mat = np.stack([v for k, v in emb.items()])
                    N, D = emb_mat.shape

                    if self.args.standardize:
                        logging.info("performing standardization ... ")
                        emb_mat = StandardScaler(with_std=True).fit(emb_mat).transform(emb_mat)

                    # if self.args.pca < D and self.args.pca > 100:
                    #     logging.info("performing PCA ... {} components".format(self.args.pca))
                    #     pca = PCA(n_components=self.args.pca)
                    #
                    #     start = time.time()
                    #     pca.fit(emb_mat)
                    #     logging.info("fitting time: {}".format(time.time() - start))
                    #
                    #     emb_mat = pca.transform(emb_mat)
                    #
                    #     var_ratio = sum(pca.explained_variance_ratio_)
                    #     logging.info('after PCA: {}, var ratio: {}'.format(emb_mat.shape, var_ratio))

                    self.mode_emb_cluster[layer][mode] = [idxs, emb_mat, cluster]


    def cluster(self, mode, layer, argument):

        idxs, emb_mat, cluster = self.mode_emb_cluster[layer][mode]
        logging.info("clustering mode: {}, layer: {} ..., th: {}".format(mode, layer, argument))

        if mode == 'r':
            linkage = self.args.r_linkage
        else:
            linkage = 'complete'
        self.clustering = AgglomerativeClustering(affinity='cosine',
                                                  linkage=linkage,
                                                  n_clusters=None,
                                                  distance_threshold=argument)

        self.clustering.fit(emb_mat)
        labels = list(self.clustering.labels_)
        clustered_labels = dict(zip(idxs, labels))

        clustered_labels = {k: {v} for k, v in clustered_labels.items()}
        if mode in ['hi', 'ti', 'hiti']:
            cluster = {k: set(v) for k, v in cluster.items()}
        else:
            cluster = {k: {v} for k, v in cluster.items()}
        # breakpoint()

        metrics = self.evaluate(true_labels=cluster, clustered_labels=clustered_labels)

        return metrics

    def cluster_and_evaluate_hierarchy(self, mode, layer):
        assert mode in ['hi', 'ti', 'hiti']
        idxs, emb_mat, cluster = self.mode_emb_cluster[layer][mode]
        logging.info("clustering mode: {}, layer: {} ...,".format(mode, layer))

        # scipy, sklearn
        self.clustering = AgglomerativeClustering(affinity='cosine',
                                                  linkage='complete',
                                                  n_clusters=None,
                                                  distance_threshold=0)
        # get whole tree and set dist = 0
        self.clustering.fit(emb_mat)


        cluster = {k: set(v) for k, v in cluster.items()}
        instance2eles = invertDict(cluster)

        children = self.clustering.children_
        distance = self.clustering.distances_

        n_instances = len(emb_mat)
        node = self.clustering.n_clusters_
        assert node == n_instances
        # breakpoint()
        node_cluster = {i: {idxs[i]} for i in range(node)}
        idx = 0
        for ch1, ch2 in children:
            node_cluster[node] = node_cluster[ch1].union(node_cluster[ch2])
            # d = distance[idx]
            idx += 1
            node += 1
        # coeff = (len(node_cluster) // 2 ) // len(instance2eles)
        # coeff = max(coeff, 1)
        instance_clusters = [v for k, v in instance2eles.items() if len(v) > 1]
        node_cluster = [clu for cluster_id, clu in node_cluster.items() if (len(clu) > 1 and len(clu) < n_instances) ]

        # node_cluster = [clu for cluster_id, clu in node_cluster.items() if (cluster_id > node-len(instance_clusters) and len(clu) > 1 and len(clu) < n_instances) ]
        return get_instance_metrics(instance_clusters, node_cluster)

    def get_emb_cluster(self, mode, layer):
        if mode in ['h', 'r', 't']:
            emb = {
                k: v for k, v in self.id2emb[layer].items() if k[-1] == mode
            }
            cluster = {
                k: v for k, v in self.id2cluster.items() if k[-1] == mode
            }
        elif mode in ['hi', 'ti']:
            emb = {
                k: v for k, v in self.id2emb[layer].items() if k[-2:] == mode
            }
            cluster = {
                k: v for k, v in self.id2cluster.items() if k[-2:] == mode
            }
            # breakpoint()
        elif mode == 'ht': # ht
            emb = {
                k: v for k, v in self.id2emb[layer].items() if (k[-1] in ['h', 't'])
            }
            cluster = {
                k: v for k, v in self.id2cluster.items() if (k[-1] in ['h', 't'])
            }
        elif mode == 'hiti':
            emb = {
                k: v for k, v in self.id2emb[layer].items() if (k[-2:] in ['hi', 'ti'])
            }
            cluster = {
                k: v for k, v in self.id2cluster.items() if (k[-2:] in ['hi', 'ti'])
            }
        else:
            print('error!!!!!')
            emb = self.id2emb
            cluster = self.id2cluster

        return emb, cluster

    def evaluate(self, true_labels, clustered_labels):
        logging.info("evaluating ...")
        metrics = get_metrics(
            clustered_labels,
            invertDict(clustered_labels),
            true_labels,
            invertDict(true_labels)
        )
        logging.info(metrics)

        return metrics




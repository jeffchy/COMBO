import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import sys
sys.path.append('../')
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
# from similarity_indices.similarity import similarity_metrics


def get_linkage(model):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = get_linkage(model)
    # Plot the corresponding dendrogram
    r = dendrogram(linkage_matrix, **kwargs)

    return linkage_matrix



# iris = load_iris()
# X = iris.data
# print(len(X))

c1 = pickle.load(open('../dataset_construction/wiki20m/save/0106165348_1641459228.640676.ti.cluster.pkl','rb'))
# setting distance_threshold=0 ensures we compute the full tree.
# c1 = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
# c1 = c1.fit(X)
# children = c1.children_
# node = c1.n_clusters_
# node_cluster = {i: {i} for i in range(node)}
# for ch1, ch2 in children:
#     node_cluster[node] = node_cluster[ch1].union(node_cluster[ch2])
#
#
#
#     node += 1
#
# breakpoint()
plt.title("Hierarchical Clustering Dendrogram")
# plot the top three levels of the dendrogram
plt.figure(figsize=(20,20))
plot_dendrogram(c1, truncate_mode="level", p=7)
plt.savefig('123.png')
plt.show()
#
# # setting distance_threshold=0 ensures we compute the full tree.
# c2 = AgglomerativeClustering(linkage='complete',affinity='cosine', distance_threshold=0, n_clusters=None)
# c2 = c2.fit(X)
# plt.title("Hierarchical Clustering Dendrogram")
# # plot the top three levels of the dendrogram
# plot_dendrogram(c2, truncate_mode="level", p=5)
# plt.xlabel("Number of points in node (or index of point if no parenthesis).")
# plt.show()

# l1 = get_linkage(c1)
# l2 = get_linkage(c2)
# metrics = similarity_metrics(l1, l2)

# ar_similarity = metrics.adjusted_rand()
# plt.plot(ar_similarity)
# plt.show()
# print(len(ar_similarity))

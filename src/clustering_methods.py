import networkx as nx
import numpy as np
import numpy.linalg as la
from sklearn.cluster import KMeans
from networkx.algorithms.community import louvain_communities
from scipy.sparse import csgraph
from networkx.algorithms import tree 

class ClusteringMethods():

    def __get_community_labels(self, number_nodes: int, communities: np.array) -> np.array:
        labels_ = np.zeros(number_nodes, dtype=int)
        for k, comm in enumerate(communities):
            for node in comm:
                labels_[node] = k
        return labels_

    def louvain_clustering(self, adj_matrix: np.array, number_clusters: int) -> np.array:
        graph = nx.from_numpy_array(adj_matrix)
        result_communities = np.array(louvain_communities(graph))
        return self.__get_community_labels(graph.number_of_nodes(), result_communities)

    def normalized_spectral_clustering(self, adj_matrix: np.array, num_clusters: int) -> np.array:
        l, U = la.eigh(csgraph.laplacian(adj_matrix, normed=True))
        kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters]) 
        labels_ = kmeans.labels_
        return labels_

    def spectral_clustering(self, adj_matrix: np.array, num_clusters: int) -> np.array: 
        D = np.diag(np.ravel(np.sum(adj_matrix,axis=1)))
        L = D - adj_matrix
        l, U = la.eigh(L)
        kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters])
        labels_ = kmeans.labels_
        return labels_

    def single_clustering(self, adj_matrix, num_clusters):
        graph = nx.from_numpy_array(adj_matrix)
        mst_graph = tree.maximum_spanning_edges(graph, algorithm="kruskal")
        edges = list(mst_graph)
        edges.sort(key=lambda tup: tup[2]['weight'])
        cutted_mst = nx.from_edgelist(edges)
        to_cut = edges[0:num_clusters - 1] 
        cutted_mst.remove_edges_from(to_cut)
        result_communities = list(nx.connected_components(cutted_mst))
        return self.__get_community_labels(graph.number_of_nodes(), result_communities)

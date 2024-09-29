
import networkx as nx
import numpy as np
import numpy.linalg as la
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.sparse import csgraph
from sklearn.cluster import SpectralClustering


class ClusteringMethods():
    def __init__(self, adj_matrix: np.array, num_clusters: int) -> None:
        self.adj_matrix = adj_matrix
        self.num_clusters = num_clusters

    def louvain_clustering(self) -> np.array:
        graph = nx.from_numpy_array(self.adj_matrix)
        comms = nx.algorithms.community.louvain_communities(graph)
        labels_ = np.zeros(graph.number_of_nodes(), dtype=int)
        for k, comm in enumerate(comms):
            for vertex in comm: 
                labels_[vertex] = k
        return labels_

    def spectral_clustering(self) -> np.array:
        clustering = SpectralClustering(n_clusters=self.num_clusters,
                                        assign_labels='kmeans',
                                        random_state=0,
                                        affinity='precomputed'
                                        ).fit(self.adj_matrix)
        return clustering.labels_

    #TO DO: check correction of this function 
    def normalized_spectral_clustering(self) -> np.array: 
        l, U = la.eigh(csgraph.laplacian(self.adj_matrix, normed=True))
        kmeans_result = KMeans(n_clusters=self.num_clusters).fit(U[:,1:self.num_clusters]) 
        return kmeans_result.labels_

    def single_clustering(self) -> np.array:
        single_result = AgglomerativeClustering(n_clusters = self.num_clusters,
                                                linkage = 'single'
                                                ).fit(self.adj_matrix)
        return single_result.labels_

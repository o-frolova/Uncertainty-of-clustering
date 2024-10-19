import networkx as nx
import numpy as np
import numpy.linalg as la
from sklearn.cluster import KMeans
from networkx.algorithms.community import louvain_communities
from scipy.sparse import csgraph
from networkx.algorithms import tree 

class ClusteringMethods():
    """
    A class for implementing various clustering methods on graph structures using adjacency matrices.

    Methods:
    --------
    louvain_clustering(adj_matrix: np.array, number_clusters: int) -> np.array:
        Applies the Louvain method for community detection on a graph represented by an adjacency matrix 
        and returns the community labels for each node.
    
    normalized_spectral_clustering(adj_matrix: np.array, num_clusters: int) -> np.array:
        Performs normalized spectral clustering on a graph represented by an adjacency matrix and 
        returns the cluster labels for each node.
        
    spectral_clustering(adj_matrix: np.array, num_clusters: int) -> np.array:
        Executes spectral clustering on a graph represented by an adjacency matrix and returns 
        the cluster labels for each node.
    
    single_clustering(adj_matrix: np.array, num_clusters: int) -> np.array:
        Implements single linkage clustering using the maximum spanning tree of the graph, represented 
        by an adjacency matrix, and returns the cluster labels for each node.
    """
    def _get_community_labels(self, number_nodes: int, communities: np.array) -> np.array:
        """
        Assigns community labels to nodes based on the identified communities.

        Parameters:
        ----------
        number_nodes : int
            The total number of nodes in the graph.
        communities : np.array
            An array of communities where each community is a list of node indices.

        Returns:
        -------
        np.array
            An array of labels corresponding to each node, where each label indicates the community 
            to which the node belongs.
        """
        labels_ = np.zeros(number_nodes, dtype=int)
        for k, comm in enumerate(communities):
            for node in comm:
                labels_[node] = k
        return labels_

    def louvain_clustering(self, adj_matrix: np.array, number_clusters: int) -> np.array:
        """
        Applies the Louvain method for community detection.

        Parameters:
        ----------
        adj_matrix : np.array
            The adjacency matrix representing the graph.
        number_clusters : int
            The desired number of clusters to form.

        Returns:
        -------
        np.array
            An array of community labels for each node in the graph.
        """
        graph = nx.from_numpy_array(adj_matrix)
        result_communities = np.array(louvain_communities(graph))
        return self._get_community_labels(graph.number_of_nodes(), result_communities)

    def normalized_spectral_clustering(self, adj_matrix: np.array, num_clusters: int) -> np.array:
        """
        Performs normalized spectral clustering.

        Parameters:
        ----------
        adj_matrix : np.array
            The adjacency matrix representing the graph.
        num_clusters : int
            The number of clusters to form.

        Returns:
        -------
        np.array
            An array of cluster labels for each node.
        """
        l, U = la.eigh(csgraph.laplacian(adj_matrix, normed=True))
        kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters]) 
        labels_ = kmeans.labels_
        return labels_

    def spectral_clustering(self, adj_matrix: np.array, num_clusters: int) -> np.array: 
        """
        Executes spectral clustering on a graph.

        Parameters:
        ----------
        adj_matrix : np.array
            The adjacency matrix representing the graph.
        num_clusters : int
            The number of clusters to form.

        Returns:
        -------
        np.array
            An array of cluster labels for each node.
        """
        D = np.diag(np.ravel(np.sum(adj_matrix,axis=1)))
        L = D - adj_matrix
        _, U = la.eigh(L)
        kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters])
        labels_ = kmeans.labels_
        return labels_

    def single_clustering(self, adj_matrix: np.array, num_clusters: int) -> np.array:
        """
        Implements single linkage clustering using the maximum spanning tree.

        Parameters:
        ----------
        adj_matrix : np.array
            The adjacency matrix representing the graph.
        num_clusters : int
            The desired number of clusters to form.

        Returns:
        -------
        np.array
            An array of cluster labels for each node based on the maximum spanning tree.
        """
        graph = nx.from_numpy_array(adj_matrix)
        mst_graph = self._get_maximum_spanning_tree(graph)
        cutted_mst = self._cut_edges(mst_graph, num_clusters)
        result_communities = list(nx.connected_components(cutted_mst))
        return self._get_community_labels(graph.number_of_nodes(), result_communities)

    def _get_maximum_spanning_tree(self, graph: nx.Graph) -> nx.Graph:
        """
        Obtains the maximum spanning tree of the given graph.

        Parameters:
        ----------
        graph : nx.Graph
            The input graph from which the maximum spanning tree is derived.

        Returns:
        -------
        nx.Graph
            The maximum spanning tree of the input graph.
        """
        mst_edges = tree.maximum_spanning_edges(graph, algorithm="kruskal", data=True)
        return nx.from_edgelist(mst_edges)

    def _cut_edges(self, mst_graph: nx.Graph, num_clusters: int) -> nx.Graph:
        """
        Cuts edges from the maximum spanning tree to create the desired number of clusters.

        Parameters:
        ----------
        mst_graph : nx.Graph
            The maximum spanning tree from which edges will be cut.
        num_clusters : int
            The number of clusters to form.

        Returns:
        -------
        nx.Graph
            The graph after edges have been cut.
        """
        edges = list(mst_graph.edges(data=True))
        edges.sort(key=lambda edge: edge[2]['weight'], reverse=True)  # Sort edges by weight (descending)
        
        edges_to_cut = edges[:num_clusters - 1]  # Keep num_clusters - 1 edges
        cutted_mst = mst_graph.copy()
        cutted_mst.remove_edges_from(edges_to_cut)
        
        return cutted_mst

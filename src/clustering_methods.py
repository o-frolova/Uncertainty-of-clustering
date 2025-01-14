import networkx as nx
import numpy as np
import numpy.linalg as la
from networkx.algorithms import tree
from networkx.algorithms.community import louvain_communities
from scipy.sparse import csgraph
from sklearn.cluster import KMeans


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

    def louvain_clustering(self, adj_matrix: np.array, _num_clusters: int) -> np.array:
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

        # Проверка на корректность графа
        if (adj_matrix < 0).any():
            raise ValueError("Adjacency matrix contains negative weights, which are not supported.")
        
        graph = nx.from_numpy_array(adj_matrix) 
        if not nx.is_weighted(graph):
            raise ValueError("The graph must be weighted.")
        if nx.is_directed(graph):
            raise ValueError("The graph must be undirected.")

        # Применяем алгоритм Луовена
        communities = louvain_communities(graph)
        
        # Преобразуем в метки
        num_clusters = len(communities)
        labels = self._get_community_labels(graph.number_of_nodes(), communities)
        
        return num_clusters, labels


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
        # l, U = la.eigh(csgraph.laplacian(adj_matrix, normed=True))
        # kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters]) 
        # labels_ = kmeans.labels_
        # return num_clusters, labels_
    
        if not (isinstance(num_clusters, int) and num_clusters > 0):
            raise ValueError("num_clusters must be a positive integer.")
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("adj_matrix must be a square matrix.")
        
        L = csgraph.laplacian(adj_matrix, normed=True)
        _, eigvecs = la.eigh(L)
        embedding = eigvecs[:, :num_clusters]
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embedding)
        labels_ = kmeans.labels_
        
        return num_clusters, labels_

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
        # D = np.diag(np.ravel(np.sum(adj_matrix,axis=1)))
        # L = D - adj_matrix
        # _, U = la.eigh(L)
        # kmeans = KMeans(n_clusters=num_clusters).fit(U[:,1:num_clusters])
        # labels_ = kmeans.labels_
        # return num_clusters, labels_
    
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("adj_matrix must be a square matrix.")
        if not np.allclose(adj_matrix, adj_matrix.T):
            raise ValueError("adj_matrix must be symmetric.")
        
        D = np.diag(np.sum(adj_matrix, axis=1))
        L = D - adj_matrix 
        
        _, eigvecs = la.eigh(L)
    
        embedding = eigvecs[:, :num_clusters] 
        
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels_ = kmeans.fit_predict(embedding)
        
        return num_clusters, labels_

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
        # graph = nx.from_numpy_array(adj_matrix)
        # mst_graph = self._get_maximum_spanning_tree(graph)
        # cutted_mst = self._cut_edges(mst_graph, num_clusters)
        # result_communities = list(nx.connected_components(cutted_mst))
        # return num_clusters, self._get_community_labels(graph.number_of_nodes(), result_communities)
        
        # from sklearn.cluster import AgglomerativeClustering
        # clustering = AgglomerativeClustering(linkage ='single', n_clusters = num_clusters).fit(adj_matrix)
        # return num_clusters, clustering.labels_
    
        # G = nx.from_numpy_array(adj_matrix)
        # mst = tree.maximum_spanning_edges(G, algorithm="kruskal")
        # edgelist = list(mst)
        # edgelist.sort(key=lambda tup: tup[2]['weight'])
        # cutted_mst = nx.from_edgelist(edgelist)
        # to_cut = edgelist[0:num_clusters - 1] 
        # cutted_mst.remove_edges_from(to_cut)

        # cc = list(nx.connected_components(cutted_mst))
        # labels_ = np.zeros(cutted_mst.number_of_nodes(), dtype=int)
        # for k, comm in enumerate(cc):
        #     for label in comm: 
        #         labels_[label] = k
        # return num_clusters, labels_


            # Построение графа из матрицы смежности
        G = nx.from_numpy_array(adj_matrix)
        
        # Нахождение максимального остовного дерева (MST)
        mst_edges = tree.maximum_spanning_edges(G, algorithm="kruskal", data=True)
        edgelist = list(mst_edges)

        # Сортировка ребер MST по убыванию веса
        edgelist.sort(key=lambda tup: tup[2]['weight'], reverse=True)

        # Удаление (num_clusters - 1) самых тяжелых ребер
        to_cut = edgelist[:num_clusters - 1]
        mst_graph = nx.Graph(edgelist)  # Восстановление графа из списка ребер
        mst_graph.remove_edges_from([edge[:2] for edge in to_cut])

        # Получение меток кластеров из связных компонент
        communities = list(nx.connected_components(mst_graph))
        labels = np.zeros(mst_graph.number_of_nodes(), dtype=int)
        for cluster_idx, community in enumerate(communities):
            for node in community:
                labels[node] = cluster_idx
        return num_clusters, labels

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

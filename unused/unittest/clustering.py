import unittest
import numpy as np
import networkx as nx
from src.clustering_methods import ClusteringMethods


class TestClusteringMethods(unittest.TestCase):

    def setUp(self):
        self.clustering_methods = ClusteringMethods()

    def test_louvain_sparse_graph(self):
        # Граф с несколькими ребрами (разреженный граф)
        adj_matrix = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0]
        ])
        num_clusters, labels = self.clustering_methods.louvain_clustering(adj_matrix, 2)
        self.assertEqual(num_clusters, 2)
        self.assertEqual(len(labels), len(adj_matrix))

    def test_normalized_spectral_dense_graph(self):
        # Полносвязный граф
        adj_matrix = np.ones((5, 5)) - np.eye(5)
        num_clusters, labels = self.clustering_methods.normalized_spectral_clustering(adj_matrix, 3)
        self.assertEqual(num_clusters, 3)
        self.assertEqual(len(labels), len(adj_matrix))

    def test_spectral_clustering_heterogeneous_graph(self):
        # Граф с двумя компонентами связности
        adj_matrix = np.array([
            [0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0]
        ])
        num_clusters, labels = self.clustering_methods.spectral_clustering(adj_matrix, 2)
        self.assertEqual(num_clusters, 2)
        self.assertEqual(len(labels), len(adj_matrix))

    def test_single_clustering_no_clusters(self):
        # Граф с равномерным шумом (отсутствие кластеризации)
        adj_matrix = np.random.rand(5, 5)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2  # Симметризация
        np.fill_diagonal(adj_matrix, 0)  # Нули на диагонали
        num_clusters, labels = self.clustering_methods.single_clustering(adj_matrix, 2)
        self.assertEqual(num_clusters, 2)
        self.assertEqual(len(labels), len(adj_matrix))

    def test_single_clustering_one_cluster(self):
        # Проверка на один кластер
        adj_matrix = np.array([
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0]
        ])
        num_clusters, labels = self.clustering_methods.single_clustering(adj_matrix, 1)
        self.assertEqual(num_clusters, 1)
        self.assertTrue(np.all(labels == 0))

    def test_clustering_more_clusters_than_nodes(self):
        # Число кластеров больше, чем количество узлов
        adj_matrix = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        with self.assertRaises(ValueError):
            self.clustering_methods.single_clustering(adj_matrix, 5)  # Некорректный запрос

    def test_empty_graph(self):
        # Проверка для пустого графа (без узлов)
        adj_matrix = np.array([])
        with self.assertRaises(ValueError):
            self.clustering_methods.single_clustering(adj_matrix, 2)

    def test_single_node_graph(self):
        # Граф с одним узлом
        adj_matrix = np.array([[0]])
        num_clusters, labels = self.clustering_methods.single_clustering(adj_matrix, 1)
        self.assertEqual(num_clusters, 1)
        self.assertEqual(labels[0], 0)


if __name__ == "__main__":
    unittest.main()

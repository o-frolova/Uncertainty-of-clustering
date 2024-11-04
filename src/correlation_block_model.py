import numpy as np

class CorrelationBlockModel:
    """
    Class for implementaition correlation block model

    Attributes:
        num_clusters (int): The number of clusters in the model.
        size_cluster (int): The number of elements in each cluster.
        r_in (float): The correlation value for elements within the same cluster.
        r_out (float): The correlation value for elements between different clusters.
    
    Methods:
        _get_covariance_matrix() -> np.ndarray:
            Constructs and returns the block covariance matrix for the clusters.
        
        _get_correlation_matrix(covariance_matrix: np.ndarray) -> np.ndarray:
            Computes and returns the correlation matrix from a given covariance matrix.
        
        _get_mean_vectors() -> np.ndarray:
            Returns a mean vector with zeros for each element in the model.
        
        _get_community_labels() -> np.ndarray:
            Returns an array of community labels for each element in the model, 
            identifying cluster membership.
        
        create_correlation_block_model() -> dict:
            Generates a complete correlation block model, returning a dictionary
            with the covariance matrix, correlation matrix, mean vector, and labels.
    """

    def __init__(self, num_clusters: int, size_cluster: int, r_in: float, r_out: float) -> None:
        """
        Initializes the CorrelationBlockModel.
        Args:
            num_clusters (int): The number of clusters in the model.
            size_cluster (int): The number of elements in each cluster.
            r_in (float): The correlation value for elements within the same cluster.
            r_out (float): The correlation value for elements between different clusters.
        """
        self.num_clusters = num_clusters
        self.size_cluster = size_cluster
        self.r_in = r_in
        self.r_out = r_out

    def _get_covariance_matrix(self) -> np.ndarray:
        """
        Constructs a block covariance matrix for the clusters based on the intra-cluster 
        and inter-cluster correlation values.

        Returns:
            np.ndarray: A 2D array representing the covariance matrix with specified 
            intra-cluster and inter-cluster correlations.
        """
        # Create the intra-cluster and inter-cluster covariance blocks
        r_ins = np.full((self.size_cluster, self.size_cluster), self.r_in)
        r_outs = np.full((self.size_cluster, self.size_cluster), self.r_out)
        # Assemble the block matrix for the full covariance structure
        cov = np.block([
            [np.tile(r_outs, k), r_ins, np.tile(r_outs, self.num_clusters - k - 1)]
            for k in range(self.num_clusters)
        ])
        # Set diagonal to 1
        np.fill_diagonal(cov, 1)
        return cov
    
    def _get_correlation_matrix(self, covariance_matrix: np.ndarray) -> np.ndarray:
        """
        Converts a covariance matrix to a correlation matrix by normalizing each element 
        with the standard deviations of its corresponding variables.

        Args:
            covariance_matrix (np.ndarray): A 2D array representing the covariance matrix.
        Returns:
            np.ndarray: A 2D array representing the corresponding correlation matrix.
        """
        v = np.sqrt(np.diag(covariance_matrix))
        outer_v = np.outer(v, v)
        correlation = covariance_matrix / outer_v
        correlation[covariance_matrix == 0] = 0
        return correlation

    def _get_mean_vectors(self) -> np.ndarray:
        """
        Generates a mean vector with zero values for each variable in the model.

        Returns:
            np.ndarray: A 1D array of zeros, with a length equal to the total number 
            of variables (num_clusters * size_cluster).
        """
        return np.zeros(self.num_clusters * self.size_cluster)

    def _get_community_labels(self) -> np.ndarray:
        """
        Assigns a community label to each variable, indicating the cluster it belongs to.

        Returns:
            np.ndarray: A 1D array with integer labels for each variable, where each 
            cluster is represented by a unique integer label.
        """
        true_labels = np.zeros(self.num_clusters * self.size_cluster, dtype=int)
        for i in range(self.num_clusters):
            true_labels[i * self.size_cluster:(i + 1) * self.size_cluster] = i
        return true_labels
    
    def create_correlation_block_model(self) -> dict:
        """
        Creates a correlation block model, including the covariance matrix, 
        correlation matrix, mean vector, and community labels.

        Returns:
            dict: A dictionary containing the following keys:
                - 'covariance_matrix' (np.ndarray): The covariance matrix for the model.
                - 'correlation_matrix' (np.ndarray): The correlation matrix for the model.
                - 'mean_vector' (np.ndarray): A mean vector of zeros.
                - 'labels' (np.ndarray): Community labels for each variable.
        """
        correlation_block_model = {}
        covariance_matrix = self._get_covariance_matrix()
        correlation_block_model['covariance_matrix'] = covariance_matrix
        correlation_block_model['correlation_matrix'] = self._get_correlation_matrix(covariance_matrix)
        correlation_block_model['mean_vector'] = self._get_mean_vectors()
        correlation_block_model['labels'] = self._get_community_labels()
        return correlation_block_model

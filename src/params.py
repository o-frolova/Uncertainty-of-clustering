
from src.clustering_methods import ClusteringMethods
from src.correlation import CorrelationMeasurement
from src.multivariate_distribution import MultivariateDistribution

cluster_method = ClusteringMethods()
correlation_method = CorrelationMeasurement()
multivariate_distribution = MultivariateDistribution()

PARAMS = {
    'clustering_method': [
                           cluster_method.single_clustering, 
                           cluster_method.louvain_clustering,
                           cluster_method.spectral_clustering,
                           cluster_method.normalized_spectral_clustering
                        ],
    'correlation_network': [
                             correlation_method.Pearson, 
                             correlation_method.Kendall,
                             correlation_method.Fechner
                           ],
    'multivariate_distribution': [
                                   multivariate_distribution.normal_distribution, 
                                   multivariate_distribution.student_distribution
                                 ],
    'number_clusters': [2, 4, 6],
    'sample_size_of_observations': [10, 20, 40, 60]
}


PARAMS_NAME = {
    'clustering_method': [
                           'single_clustering', 
                           'louvain_clustering',
                           'spectral_clustering',
                           'normalized_spectral_clustering'
                         ],
    'correlation_network': [
                             'Pearson', 
                             'Kendall',
                             'Fechner'
                           ],
    'multivariate_distribution': [
                                   'normal_distribution', 
                                   'student_distribution'
                                 ],
    'number_clusters': [2, 4, 6],
    'sample_size_of_observations': [10, 20, 40, 60]

}

from src.multivariate_distribution import MultivariateDistribution 
from src.correlation import CorrelationMeasurement 
from src.clustering_methods import ClusteringMethods
import numpy as np 

class CorrelationBlockModel():

    def _generated_data(self,
                       multivariate_distribution: MultivariateDistribution,
                       mean_vector: np.array,
                       cov_matrix: np.array,
                       sample_size_of_observations: int
    ) -> np.array:
        generated_vectors = multivariate_distribution(mean_vector = mean_vector,
                                                      cov_matrix = cov_matrix,
                                                      num_generated_samples = sample_size_of_observations
                                                     )  
        
        observations_random_variables = []
        for random_value_index in range(len(generated_vectors[0])):
            observations_random_variables.append(generated_vectors[:, random_value_index])

        return np.array(observations_random_variables)
    
    def _correlation_estimation(self,
                               observations_random_variables: np.array,
                               correlation_method: CorrelationMeasurement
    ) -> np.array:
        correlation_matrix = []
        for observation_random_variable_1 in observations_random_variables:
            row = []
            for observation_random_variable_2 in observations_random_variables:
                row.append(correlation_method(observation_random_variable_1, observation_random_variable_2))
            correlation_matrix.append(row)
        return np.array(correlation_matrix)
    
    def _clustering(self,
                          correlation_estimation: np.array,
                          clustering_method: ClusteringMethods,
                          number_clusters: int
    ) -> np.array:
        return clustering_method(correlation_estimation, number_clusters)
    
    def clustering(self,
                    multivariate_distribution: MultivariateDistribution,
                    mean_vector: np.array,
                    cov_matrix: np.array,
                    sample_size_of_observations: int,
                    correlation_method: CorrelationMeasurement,
                    clustering_method: ClusteringMethods,
                    number_clusters: int
    ) -> np.array:
       
        observations_random_variables = self._generated_data(multivariate_distribution = multivariate_distribution,
                                              mean_vector = mean_vector,
                                              cov_matrix = cov_matrix,
                                              sample_size_of_observations = sample_size_of_observations
                                              )
        correlation_estimation = self._correlation_estimation(observations_random_variables = observations_random_variables,
                                                              correlation_method = correlation_method
                                                              )
        
        return self._clustering(correlation_estimation = correlation_estimation,
                                clustering_method = clustering_method,
                                number_clusters = number_clusters)

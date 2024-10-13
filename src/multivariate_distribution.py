import numpy as np

class MultivariateDistribution():
    
    def normal_distribution(self, mean_vector: np.array, cov_matrix: np.array, num_generated_samples: int) -> np.array:
        return np.random.multivariate_normal(mean = mean_vector,
                                             cov = cov_matrix, 
                                             size = num_generated_samples)
    
    def student_distribution(self, mean_vector: np.array, cov_matrix: np.array, num_generated_samples: int, degree_freedom: int = 3) -> np.array:
        x = np.random.chisquare(degree_freedom, num_generated_samples) / degree_freedom
        z = np.random.multivariate_normal(np.zeros(len(mean_vector)), cov_matrix, (num_generated_samples,))
        res =  mean_vector + z/np.sqrt(x)[:,None]
        return res

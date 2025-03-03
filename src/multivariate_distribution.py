import numpy as np


class MultivariateDistribution():
    """
    A class for generating random samples from multivariate probability distributions.

    Methods:
    --------
    normal_distribution(mean_vector: np.array, cov_matrix: np.array, num_generated_samples: int) -> np.array:
        Generates samples from a multivariate normal distribution based on the specified mean vector 
        and covariance matrix.
        
    student_distribution(mean_vector: np.array, cov_matrix: np.array, num_generated_samples: int, 
                         degree_freedom: int = 3) -> np.array:
        Generates samples from a multivariate Student's t-distribution, utilizing a combination 
        of chi-squared and multivariate normal distributions, with adjustable degrees of freedom.
    """
    
    def normal_distribution(
            self,
            mean_vector: np.array,
            cov_matrix: np.array,
            num_generated_samples: int
    ) -> np.array:
        """
        Generates a sample from a multivariate normal distribution.

        Parameters:
        ----------
        mean_vector : np.array
            The mean vector of the multivariate normal distribution.
        cov_matrix : np.array
            The covariance matrix representing the relationships between variables.
        num_generated_samples : int
            The number of samples to generate.

        Returns:
        -------
        np.array
            An array of shape (num_generated_samples, len(mean_vector)) containing 
            the generated samples from the multivariate normal distribution.
        """
        return np.random.multivariate_normal(mean = mean_vector,
                                             cov = cov_matrix, 
                                             size = num_generated_samples)
    
    def student_distribution(
            self,
            mean_vector: np.array,
            cov_matrix: np.array,
            num_generated_samples: int,
            degree_freedom: int = 3
    ) -> np.array:
        """
        Generates a sample from a multivariate Student's t-distribution.
       
        This function generates samples using a combination of the chi-squared distribution 
        and multivariate normal distribution. The chi-squared distribution accounts for the 
        variability introduced by the degrees of freedom, while the multivariate normal 
        distribution represents the underlying correlations between the variables.

        Process:
        --------
        1. A chi-squared sample (`x`) is generated with the specified degrees of freedom, and 
        each value is normalized by the degrees of freedom.
        - `x = np.random.chisquare(degree_freedom, num_generated_samples) / degree_freedom`
        This step introduces variability in the scaling, accounting for the distribution's 
        "fat tails."
        
        2. A sample from a multivariate normal distribution (`z`) is generated with a mean vector 
        of zeros and the specified covariance matrix.
        - `z = np.random.multivariate_normal(np.zeros(len(mean_vector)), cov_matrix, (num_generated_samples,))`
        This sample reflects the correlations between the variables but has zero mean.
        
        3. The final sample is generated by adjusting the normal sample using the square root 
        of the chi-squared sample and adding the mean vector.
        `mean_vector + z / np.sqrt(x)[:, None]`
        This scaling step adjusts the normal sample to follow a Student's t-distribution, accounting 
        for small degrees of freedom, and shifts the result by the specified mean vector.

        Parameters:
        ----------
        mean_vector : np.array
            The mean vector of the multivariate Student's t-distribution.
        cov_matrix : np.array
            The covariance matrix representing the relationships between variables.
        num_generated_samples : int
            The number of samples to generate.
        degree_freedom : int, optional
            Degrees of freedom for the Student's t-distribution. Default is 3.

        Returns:
        -------
        np.array
            An array of shape (num_generated_samples, len(mean_vector)) containing 
            the generated samples from the multivariate Student's t-distribution.
        """
        x = np.random.chisquare(degree_freedom, num_generated_samples) / degree_freedom
        z = np.random.multivariate_normal(np.zeros(len(mean_vector)), cov_matrix, (num_generated_samples,))
        return mean_vector + z/np.sqrt(x)[:,None]

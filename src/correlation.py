import numpy as np
from scipy import stats


class CorrelationMeasurement:
    """
    A class to calculate various correlation coefficients between two data sets.

    Methods:
    --------
    Pearson(data_1: np.array, data_2: np.array) -> float:
        Returns the absolute value of Pearson correlation coefficient for two data sets.
    
    Kendall(data_1: np.array, data_2: np.array) -> float:
        Returns the absolute value of Kendall correlation coefficient, derived through
        a transformation of the Pearson correlation (for elliptical distributions).
        
    Fechner(data_1: np.array, data_2: np.array) -> float:
        Returns the absolute value of Fechner correlation coefficient, based on
        sign comparison of deviations of two data sets from their respective means.
    """
    
    def Pearson(self, data_1: np.array, data_2: np.array) -> float:
        """
        Calculates the absolute value of Pearson correlation coefficient for two data sets.

        Pearson correlation measures the linear relationship between two data sets, 
        but it assumes the data follows a normal distribution. Hence, the Pearson method is 
        suitable for data with linear dependence.

        Parameters:
        -----------
        data_1 : np.array
            Observations of the first random variable
        data_2 : np.array
            Observations of the second random variable

        Returns:
        --------
        float
            The absolute value of Pearson correlation coefficient.
        """
        return abs(stats.pearsonr(data_1, data_2).statistic)
    
    def Kendall(self, data_1: np.array, data_2: np.array) -> float:
        """
        Calculates the absolute value of Kendall correlation coefficient using a transformation
        of the Pearson correlation.

        Kendall's tau is typically used for ordinal data or non-parametric statistics. In elliptical distributions, 
        it can be approximately related to Pearson correlation via a trigonometric transformation: tau ≈ (2 / π) * arcsin(Pearson).
        This method assumes the data has a continuous and elliptical distribution.

        Parameters:
        -----------
        data_1 : np.array
            Observations of the first random variable
        data_2 : np.array
            Observations of the second random variable

        Returns:
        --------
        float
            The absolute value of Kendall correlation coefficient, derived from Pearson's correlation.
        """
        return abs((2 / np.pi) * np.arcsin(self.Pearson(data_1, data_2)))
        
    def Fechner(self, data_1: np.array, data_2: np.array) -> float:
        """
        Calculates the absolute value of Fechner correlation coefficient for two data sets.

        Fechner correlation uses the signs of deviations from the mean to determine whether the two variables 
        move together or in opposite directions. This method counts how often the deviations from the mean 
        have the same or opposite sign, making it useful for detecting monotonic trends without assuming linearity.

        Parameters:
        -----------
        data_1 : np.array
            Observations of the first random variable
        data_2 : np.array
            Observations of the second random variable

        Returns:
        --------
        float
            The absolute value of Fechner correlation coefficient, based on sign comparison
            of the deviations of the data sets from their means.
        """
        data_1_div = data_1 - np.mean(data_1)
        data_2_div = data_2 - np.mean(data_2)
        return abs(np.sum(np.sign(data_1_div * data_2_div)) / data_1.shape[0])

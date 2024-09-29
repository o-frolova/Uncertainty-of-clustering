from scipy import stats
import numpy as np 
from scipy.stats._result_classes import PearsonRResult, SignificanceResult

class CorrelationMeasurement():

    def __init__(self, data_1: np.array, data_2: np.array) -> None:
        self.data_1 = data_1
        self.data_2 = data_2
        self.__check_data()
    
    def __check_data(self):
        if len(self.data_1) != len(self.data_2):
            raise ValueError("Both inputs must have the same length")
        if not (isinstance(self.data_1, np.array) and isinstance(self.data_2, np.array)):
            raise TypeError("Input data must be np.array")

    def Pearson(self) -> PearsonRResult:
        return stats.pearsonr(self.data_1, self.data_2)

    def Kendall(self) -> SignificanceResult:
        return stats.kendalltau(self.data_1, self.data_2 )

    def Fechner(self) -> float:
        data_1_div = self.data_1 - np.mean(self.data_1)
        data_2_div = self.data_2 - np.mean(self.data_2)
        return np.sum(np.sign(data_1_div * data_2_div)) / self.data_1.shape[0], 0

from scipy import stats
import numpy as np 


class CorrelationMeasurement():
    
    def Pearson(self, data_1: np.array, data_2: np.array) -> float:
        return abs(stats.pearsonr(data_1, data_2).statistic)
    
    def Kendall(self, data_1: np.array, data_2: np.array) -> float:
        return abs(stats.kendalltau(data_1, data_2).statistic)
        
    def Fechner(self, data_1: np.array, data_2: np.array) -> float:

        data_1_div = data_1 - np.mean(data_1)
        data_2_div = data_2 - np.mean(data_2)

        return abs(np.sum(np.sign(data_1_div * data_2_div)) / data_1.shape[0])

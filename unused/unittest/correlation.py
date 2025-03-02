import unittest
import numpy as np
from scipy import stats

from src.correlation import CorrelationMeasurement


class TestCorrelationMeasurement(unittest.TestCase):
    def setUp(self):
        # Инициализация данных для тестов
        self.data_1 = np.array([1, 2, 3, 4, 5])
        self.data_2 = np.array([2, 4, 6, 8, 10])  # Линейно зависимые от data_1
        self.data_3 = np.array([5, 4, 3, 2, 1])  # Обратная зависимость от data_1
        self.data_4 = np.array([10, 9, 8, 7, 6])
        self.data_5 = np.array([1, 1, 1, 1, 1])  # Массив с одинаковыми значениями
        self.correlation = CorrelationMeasurement()

    def test_Pearson(self):
        # Тестирование метода Pearson
        pearson_value = self.correlation.Pearson(self.data_1, self.data_2)
        expected_value, _ = stats.pearsonr(self.data_1, self.data_2)
        self.assertAlmostEqual(pearson_value, expected_value, places=1)

    def test_Kendall(self):
        # Тестирование метода Kendall
        kendall_value = self.correlation.Kendall(self.data_1, self.data_2)
        res = stats.kendalltau(self.data_1, self.data_2)
        expected_value = res.statistic
        self.assertAlmostEqual(kendall_value, expected_value, places=1)

    def test_Fechner_positive_correlation(self):
        # Тестирование метода Fechner при положительной корреляции
        fechner_value = self.correlation.Fechner(self.data_1, self.data_2)
        self.assertAlmostEqual(fechner_value, 1.0, places=1)  # Полная положительная корреляция

    def test_Fechner_negative_correlation(self):
        # Тестирование метода Fechner при отрицательной корреляции
        fechner_value = self.correlation.Fechner(self.data_1, self.data_4)
        self.assertAlmostEqual(fechner_value, -1.0, places=1)  # Полная отрицательная корреляция

    def test_Fechner_no_correlation(self):
        # Тестирование метода Fechner при отсутствии корреляции
        random_data_1 = np.array([1, 2, 3, 4, 5])
        random_data_2 = np.array([5, 4, 3, 2, 1])
        fechner_value = self.correlation.Fechner(random_data_1, random_data_2)
        self.assertLess(fechner_value, 1.0)  # Не должно быть полной корреляции

    # def test_empty_data(self):
    #     # Тестирование метода на пустых данных
    #     empty_data_1 = np.array([])
    #     empty_data_2 = np.array([])
    #     with self.assertRaises(ValueError):  # Ожидаем, что будет вызвана ошибка
    #         self.correlation.Pearson(empty_data_1, empty_data_2)

    # def test_data_with_nan(self):
    #     # Тестирование метода на данных с NaN
    #     data_with_nan_1 = np.array([1, 2, 3, np.nan, 5])
    #     data_with_nan_2 = np.array([2, 4, 6, 8, np.nan])
    #     with self.assertRaises(ValueError):  # Ожидаем ошибку или обработку NaN значений
    #         self.correlation.Pearson(data_with_nan_1, data_with_nan_2)

    # def test_identical_data(self):
    #     # Тестирование метода на одинаковых данных (корреляция должна быть равна 1.0)
    #     identical_data = np.array([5, 5, 5, 5, 5])
    #     pearson_value = self.correlation.Pearson(identical_data, identical_data)
    #     expected_value, _ = stats.pearsonr(identical_data, identical_data)
    #     self.assertEqual(pearson_value, expected_value)
    #     # self.assertEqual(pearson_value, 1.0)

    # def test_zero_variance(self):
    #     # Тестирование метода на данных с нулевой дисперсией (все значения одинаковы)
    #     zero_variance_data = np.array([2, 2, 2, 2, 2])
    #     with self.assertRaises(ValueError):  # Корреляция не определена для данных с нулевой дисперсией
    #         self.correlation.Pearson(zero_variance_data, self.data_1)

    def test_random_data(self):
        # Тестирование метода на случайных данных (корреляция должна быть близка к нулю)
        random_data_1 = np.random.random(100)
        random_data_2 = np.random.random(100)
        fechner_value = self.correlation.Fechner(random_data_1, random_data_2)
        self.assertTrue(-1 <= fechner_value <= 1)  # Проверка, что результат в допустимом диапазоне

if __name__ == '__main__':
    unittest.main()

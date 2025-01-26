import ast
import pandas as pd
import pathlib

from tap import Tap
from src.params import PARAMS_NAME

class CustomParser(Tap):
    data_path: pathlib.Path
    path_to_save_folder: pathlib.Path
    base_name_file: str

class ConvertTable:
    def __init__(self, args):
        self.data_path = args.data_path
        self.path_to_save_folder = args.path_to_save_folder
        self.base_name_file = args.base_name_file

    def convert_table(self) -> None:
        data = pd.read_csv(self.data_path)
        clustering_order = PARAMS_NAME['clustering_method']
        
        for multi_distribution in PARAMS_NAME['multivariate_distribution']:
            for corr_network in PARAMS_NAME['correlation_network']:
                    for size_samples in PARAMS_NAME['sample_size_of_observations']:
                        
                        filtered_data = data[(data['correlation_network'] == corr_network) & 
                                            (data['multivariate_distribution'] == multi_distribution) & 
                                            (data['sample_size_of_observations'] == size_samples) &
                                            (data['clustering_method'] != 'louvain_clustering')]
                        
                        filtered_data['ARI'] = filtered_data['ARI'].round(2)
                        
                        filtered_data = filtered_data.drop(['cluster_distribution'], axis=1)

                        if filtered_data.empty:
                            continue

                        filtered_data = filtered_data[filtered_data['clustering_method'].isin(clustering_order)]
                        
                        filtered_data['clustering_method'] = pd.Categorical(
                            filtered_data['clustering_method'], 
                            categories=clustering_order, 
                            ordered=True
                        )

                        result = pd.pivot_table(
                            filtered_data, 
                            values = 'ARI', 
                            index = 'clustering_method', 
                            columns = 'number_clusters', 
                            aggfunc = 'first',
                            sort = False,
                            observed = False
                        )
                        
                        file_name = f'{self.base_name_file}_{multi_distribution}_{corr_network}_{size_samples}.csv'
                        result.to_csv(self.path_to_save_folder + file_name)

                        filtered_data = data[(data['correlation_network'] == corr_network) & 
                                            (data['multivariate_distribution'] == multi_distribution) & 
                                            (data['sample_size_of_observations'] == size_samples) &
                                            (data['clustering_method'] == 'louvain_clustering')]
                        filtered_data['cluster_distribution'] = filtered_data['cluster_distribution'].map(ast.literal_eval)

                        def get_statistics(x):
                            return pd.DataFrame(x[1]).describe().T

                        info_stat = filtered_data['cluster_distribution'].map(get_statistics)

                        filtered_data = filtered_data.drop(['cluster_distribution'], axis=1)
                        result_louvain = pd.concat([filtered_data, info_stat.apply(lambda x: x.squeeze() if isinstance(x, pd.DataFrame) else x)], axis=1)
                        result_louvain = result_louvain.drop(['Unnamed: 0', 'count', 'correlation_network', 'multivariate_distribution', 'sample_size_of_observations'], axis=1)
                        result_louvain['ARI'] = result_louvain['ARI'].round(2)
                        result_louvain['mean'] = result_louvain['mean'].round(2)
                        result_louvain['std'] = result_louvain['std'].round(2)
                        file_name = f'{self.base_name_file}_{multi_distribution}_{corr_network}_{size_samples}_louvain.csv'
                        result_louvain.to_csv(self.path_to_save_folder + file_name)


if __name__ == "__main__":
    ARGS = CustomParser(underscores_to_dashes=True).parse_args()
    convert = ConvertTable(ARGS)
    convert.convert_table()
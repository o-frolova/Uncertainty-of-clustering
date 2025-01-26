import os
import pathlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tap import Tap
from typing import List

class CustomParser(Tap):
    folder_path: pathlib.Path
    correlation_network: str
    multivariate_distribution: str
    sample_size_of_observations: int
    number_clusters: int
    save_path: pathlib.Path
    max_r_out: float
    step: float

class LineGraphDrawer:
    def __init__(self, args: Tap):
        self.folder_path = args.folder_path
        self.correlation_network = args.correlation_network
        self.multivariate_distribution = args.multivariate_distribution
        self.sample_size_of_observations = args.sample_size_of_observations
        self.number_clusters = args.number_clusters
        self.save_path = args.save_path
        self.max_r_out = args.max_r_out
        self.step = args.step
        self.files = self.get_files()

    def get_files(self) -> List[str]:
        files = []
        for file_name in os.listdir(self.folder_path):
            if os.path.isfile(os.path.join(self.folder_path, file_name)):
                files.append(os.path.join(self.folder_path, file_name))
        return sorted(files)

    def extractor_scores(self) -> dict:
        clustering_methods = [
            'single_clustering',
            'louvain_clustering',
            'spectral_clustering',
            'normalized_spectral_clustering'
        ]
        scores = {method: [] for method in clustering_methods}
        
        for file in self.files:
            df = pd.read_csv(file)
            for method in clustering_methods:
                df_filtered = df[(df['clustering_method'] == method) &
                                 (df['correlation_network'] == self.correlation_network) &
                                 (df['multivariate_distribution'] == self.multivariate_distribution) &
                                 (df['sample_size_of_observations'] == self.sample_size_of_observations) &
                                 (df['number_clusters'] == self.number_clusters)]
                if not df_filtered.empty:
                    scores[method].append(float(df_filtered['ARI']))
                else:
                    scores[method].append(0)  # Default value if no data matches
        return scores

    def visual_iteration_r_out(self, data: dict) -> None:
        r_out_values = np.arange(0, self.max_r_out + self.step, self.step)
        plt.figure(figsize=(10, 6))

        for method, scores in data.items():
            plt.plot(r_out_values, scores, label=method, linestyle='-')

        plt.title('Dependence of Metric on r_out', fontsize=16)
        plt.xlabel('r_out', fontsize=14)
        plt.ylabel('Metric Value (ARI)', fontsize=14)
        plt.legend(title="Clustering Methods", fontsize=12)
        plt.grid(alpha=0.5)
        plt.tight_layout()

        # Save plot
        plt.savefig(self.save_path, format='png')

    def draw_graph(self):
        scores = self.extractor_scores()
        self.visual_iteration_r_out(scores)

if __name__ == "__main__":
    ARGS = CustomParser(underscores_to_dashes=True).parse_args()
    drawer = LineGraphDrawer(ARGS)
    drawer.draw_graph()

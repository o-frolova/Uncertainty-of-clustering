import os
import pathlib
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
from tap import Tap


class CustomParser(Tap):
    folder_path: pathlib.Path
    correlation_network: str
    multivariate_distribution: str
    sample_size_of_observations: int
    number_clusters: int
    save_path: pathlib.Path
    max_r_out: float
    step: float


class BarGraphDrawer:
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
        sorted_files = sorted(files)
        return sorted_files

    def extractor_scores(self) -> dict:
        routs = {
            "single_clustering": [],
            "louvain_clustering": [],
            "spectral_clustering": [],
            "normalized_spectral_clustering": [],
        }
        for file in self.files:
            df = pd.read_csv(file)
            for clustering_method in [
                "single_clustering",
                "louvain_clustering",
                "spectral_clustering",
                "normalized_spectral_clustering",
            ]:
                df_rout = df[
                    (df["clustering_method"] == clustering_method)
                    & (df["correlation_network"] == self.correlation_network)
                    & (
                        df["multivariate_distribution"]
                        == self.multivariate_distribution
                    )
                    & (
                        df["sample_size_of_observations"]
                        == self.sample_size_of_observations
                    )
                    & (df["number_clusters"] == self.number_clusters)
                ]
                print(df_rout)
                print(df_rout["ARI"])
                routs[clustering_method].append(float(df_rout["ARI"]))
        return routs

    def visual_iteration_r_out_barchart(self, data: dict) -> None:
        categories = list(data.keys())
        values = list(data.values())

        x = range(len(categories))  # Categories on x-axis
        bar_width = 0.25

        # Define positions for each group
        num_files = len(values[0])  # Number of files
        positions = [i - bar_width * (num_files - 1) / 2 for i in x]

        colors = ["#4e79a7", "#f28e2c", "#76b7b2"]
        labels = ["n = 2", "n = 2.5", "n = 3"]

        plt.figure(figsize=(12, 7))

        for idx in range(num_files):
            bars = plt.bar(
                [p + idx * bar_width for p in positions],
                [v[idx] for v in values],
                bar_width,
                label=labels[idx],
                color=colors[idx % len(colors)],
                edgecolor="black",
            )
            # Add text on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    f"{height:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="black",
                )

        # Customize plot appearance
        plt.xticks(x, categories, rotation=20, ha="right", fontsize=12)
        plt.xlabel("Clustering Method", fontsize=14, labelpad=10)
        plt.ylabel("ARI Metric Value", fontsize=14, labelpad=10)
        plt.title(
            "Comparison of Clustering Methods by ARI Values",
            fontsize=16,
            pad=20,
            weight="bold",
        )
        plt.legend(title="Degree of Freedom (n)", fontsize=12, title_fontsize=12)
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        # Save plot
        plt.savefig(self.save_path, format="png")

    def draw_graph(self):
        scores = self.extractor_scores()
        self.visual_iteration_r_out_barchart(scores)


if __name__ == "__main__":
    ARGS = CustomParser(underscores_to_dashes=True).parse_args()
    drawer = BarGraphDrawer(ARGS)
    drawer.draw_graph()

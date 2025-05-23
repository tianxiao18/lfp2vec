import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
import os


class PCAVisualizer:
    def __init__(self, label_json, output_path):
        self.output_path = output_path
        self.label_json = label_json

    def create_pca(self, embeddings, labels, n_components, session, stage, color_map=None):
        if len(str(labels[0])) == 1:
            labels = [self.label_json[str(label)] for label in labels]
        pca = PCA(n_components=n_components)
        reduced_embeddings = pca.fit_transform(embeddings)

        df = pd.DataFrame(reduced_embeddings, columns=[f"PC{i + 1}" for i in range(n_components)])
        df["Label"] = labels
        if not color_map:
            color_map = {
                "Cortex": "#f0553b", "CA1": "#ffa15a",
                "CA2": "#636efa", "CA3": "#00cc96", "DG": "#ab63fa",
                "Basal_Ganglia": "#19d3f3", "Suppl_Motor_Area": "#e763fa",
                "Primary_Motor_Cortex": "#f0b400"
            }

        unique_labels = df["Label"].astype(str).unique()
        full_color_map = {
            label: next((color for key, color in color_map.items() if key.lower() in label.lower()), "#999999")
            for label in unique_labels
        }

        if n_components == 2:
            fig = px.scatter(
                df,
                x="PC1",
                y="PC2",
                color="Label",
                color_discrete_map=full_color_map,
                title=f"2D PCA of {stage} Embeddings (Session: {session})",
                labels={"Label": "Class Label"}
            )
        elif n_components == 3:
            fig = px.scatter_3d(
                df,
                x="PC1",
                y="PC2",
                z="PC3",
                color="Label",
                color_discrete_map=full_color_map,
                title=f"3D PCA of {stage} Embeddings (Session: {session})",
                labels={"Label": "Class Label"},
            )
        else:
            raise ValueError("n_components must be 2 or 3 for visualization.")

        output_file_path = os.path.join(self.output_path,
                                        f'session_{session}_{stage}_pca_{n_components}d_visualization.html')
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        fig.write_html(output_file_path)
        print(f"PCA visualization for {stage} embeddings (Session: {session}) saved at {output_file_path}")

    def create_combined_pca(self, datasets, n_components, session, title="combined", color_map=None):
        combined_embeddings = []
        combined_labels = []
        for stage, data in datasets.items():
            combined_embeddings.append(data["embeddings"])
            if title == "combined":
                combined_labels.extend([f"{stage}_{self.label_json[str(label)[-1]]}"
                                        for label in data["labels"]])
            else:
                combined_labels.extend([f"{stage}_{label}" for label in data["labels"]])

        combined_embeddings = np.vstack(combined_embeddings)

        self.create_pca(combined_embeddings, combined_labels, n_components, session, title, color_map)

"""
Offline phase of DStream algorithm.
"""


import numpy as np
from baseframework.offline_clustering.adjacency import adjacency_clustering
from baseframework.stream_clustering import OfflinePhase_Clusterer

class DStreamOfflinePhase(OfflinePhase_Clusterer):
    def __init__(self, grid_size):
        super().__init__()
        self.grid_size = grid_size
        # The threshold for adjacency (usually grid_size or slightly larger)
        self.similarity_threshold = (grid_size / 2) * 1.1 

    def predict_summaries(self, summaries: dict) -> dict:
        """Predicts cluster labels for the grid summaries."""
        if not summaries:
            return {}
        
        # similar to how it is in the jupyter notebook
        # 1. Convert the dictionary of summaries into arrays expected by adjacency.py
        # Loop through dictionary, calculate actual X/Y from the grid indices
        # Append to centers, append status to statuses

        summary_centers = []
        statuses = []
        grid_indices_list = []


        for grid_idx, cell in summaries.items():
            # track the grid indices so we can map them to labels later!
            grid_indices_list.append(grid_idx)
            # calculate the center of the grid cell
            x_center = grid_idx[0] * self.grid_size + (self.grid_size / 2)
            y_center = grid_idx[1] * self.grid_size + (self.grid_size / 2)
            summary_centers.append([x_center, y_center])

            statuses.append(cell.status.value)
        
        
        # 2. Call the provided adjacency clustering
        print("   [Offline] Running Adjacency Clustering on summaries...")
        labels = adjacency_clustering(
            X=np.array(summary_centers),
            metric='euclidean',
            status=np.array(statuses),
            similarity_thresholds=self.similarity_threshold
        )

        clustering_dict = {idx: label for idx, label in zip(grid_indices_list, labels)}
        return clustering_dict
    
    def predict_assignment(self, assignments: list, summaries: dict) -> list[int]:
        """Maps raw data points to their corresponding cluster label."""
        summary_clustering = self.predict_summaries(summaries)
        return [summary_clustering.get(a, -1) for a in assignments]
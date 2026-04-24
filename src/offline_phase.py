"""
Offline phase of DStream algorithm.
Post-processing of summaries using DBSCAN/Adjacency Offline Clustering.
"""


import numpy as np
from baseframework.offline_clustering.adjacency import adjacency_clustering

class OfflinePhase_Clusterer:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        # The threshold for adjacency (usually grid_size or slightly larger)
        self.similarity_threshold = (grid_size / 2) * 1.1 

    def predict_summaries(self, summaries):
        """Predicts cluster labels for the grid summaries."""
        if not summaries:
            return []
        
        # similar to how it is in the jupyter notebook
        # 1. Convert the dictionary of summaries into arrays expected by adjacency.py
        summary_centers = []
        statuses = []
        
        # Loop through dictionary, calculate actual X/Y from the grid indices
        # Append to centers, append status to statuses
        
        # 2. Call the provided adjacency clustering
        labels = adjacency_clustering(
            X=np.array(summary_centers),
            metric='euclidean',
            status=np.array(statuses),
            similarity_thresholds=self.similarity_threshold
        )
        return labels, summary_centers
"""
DStream: Density-based clustering for evolving data streams.
Main implementation combining online and offline phases
"""

import numpy as np
from .online_phase import DStreamOnlinePhase
from .offline_phase import DStreamOfflinePhase

class DStream:
    def __init__(self, grid_size=0.1, decay_factor=0.998, dense_threshold=3.0, sparse_threshold=0.5, cleanup_interval=2):
        self.grid_size = grid_size
        self.online_phase = DStreamOnlinePhase(
            grid_size, decay_factor, dense_threshold, sparse_threshold, cleanup_interval, beta=0.3
        )
        self.offline_phase = DStreamOfflinePhase(grid_size)

    def learn(self, xs, ts) -> list:
        """Feed a new batch of data into the online phase using the framework's learn method."""
        return self.online_phase.learn_one(xs, ts)

    def predict(self, X_batch=None):
        """Trigger the offline phase to get the current clustering."""
        print("   [Offline] Fetching summaries from Online phase...")
        summaries = self.online_phase.get_summary()
        if X_batch is not None:
            # calculate the assignments (Grid Cell IDs) for the raw points
            grid_indices = np.floor(X_batch / self.grid_size).astype(int)
            assignments = [tuple(idx) for idx in grid_indices]
            
            # Pass the assignments to the framework's predict_assignment method
            point_labels = self.offline_phase.predict_assignment(assignments, summaries)
            return np.array(point_labels)
        else:
            # Return the raw summary clustering map if no batch is provided
            return self.offline_phase.predict_summaries(summaries)
        
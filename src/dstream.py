"""
DStream: Density-based clustering for evolving data streams.
Main implementation combining online and offline phases
"""

from .online_phase import OnlinePhase_StreamClusterer
from .offline_phase import OfflinePhase_Clusterer

class DStream:
    def __init__(self, grid_size=0.1, decay_factor=0.998, dense_threshold=3.0, sparse_threshold=0.5):
        self.online_phase = OnlinePhase_StreamClusterer(
            grid_size, decay_factor, dense_threshold, sparse_threshold
        )
        self.offline_phase = OfflinePhase_Clusterer(grid_size)

    def update(self, batch):
        """Feed a new batch of data into the online phase."""
        self.online_phase.process_batch(batch)
        self.online_phase.update_status()
        
        # TODO - optional: Implement the periodic cleanup (e.g., every T steps) here
        # to delete completely dead 'sparse' cells from the dictionary.

    def predict(self):
        """Trigger the offline phase to get the current clustering."""
        summaries = self.online_phase.get_summary()
        labels, centers = self.offline_phase.predict_summaries(summaries)
        return labels, centers
"""
DStream: Density-based clustering for evolving data streams.
Main implementation combining online and offline phases
"""

from src.online_phase import DStreamOnline
from src.offline_phase import DStreamOffline


class DStream:
    """
    DStream algorithm for stream clustering.

    Combines online phase (maintains micro-clusters) with offline phase
    (post-processes summaries using density-based clustering).
    """

    def __init__(self, **kwargs):
        """Initialize DenStream with online and offline phases."""
        self.online = DStreamOnline(**kwargs)
        self.offline = DStreamOffline(**kwargs)

    def partial_fit(self, X, **kwargs):
        """Process a batch of streaming data."""
        self.online.partial_fit(X, **kwargs)

    def fit(self, X, **kwargs):
        """Fit the algorithm to data (for batch processing)."""
        self.partial_fit(X, **kwargs)

    def predict(self, **kwargs):
        """Return final clustering."""
        micro_clusters = self.online.get_clusters()
        self.offline.cluster(micro_clusters)
        return self.offline.get_final_clusters()

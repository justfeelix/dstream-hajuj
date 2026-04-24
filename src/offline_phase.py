"""
Offline phase of DStream algorithm.
Post-processing of summaries using DBSCAN/Adjacency Offline Clustering.
"""


class DStreamOffline:
    """Offline phase of DStream algorithm."""

    def __init__(self, **kwargs):
        """Initialize offline phase parameters """
        pass

    def cluster(self, micro_clusters):
        """Cluster micro-clusters using DBSCAN or adjacency-based clustering."""
        pass

    def get_final_clusters(self):
        """Return final macro-clusters."""
        pass

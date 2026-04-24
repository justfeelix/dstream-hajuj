"""
Online phase of DStream algorithm.
Maintains core micro-clusters (CMC) and outlier micro-clusters (OMC).
"""

class CoreMicroCluster:
    """Represents a core micro-cluster in the online phase."""
    pass


class OutlierMicroCluster:
    """Represents an outlier micro-cluster in the online phase."""
    pass


class DStreamOnline:
    """Online phase of DStream algorithm."""

    def __init__(self, **kwargs):
        """Initialize online phase parameters."""
        pass

    def partial_fit(self, X, **kwargs):
        """Process a batch or stream of points."""
        pass

    def get_clusters(self):
        """Return current micro-clusters."""
        pass

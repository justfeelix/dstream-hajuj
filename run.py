"""
Main entry point for DStream experiments.
"""

import sys
sys.path.insert(0, '.')

from src.denstream import DStream
from baseframework.datahandler import load_data
from baseframework.metrics import evaluate_clustering


def main():
    """Run DStream on a dataset."""
    # Load data from baseframework
    # X = load_data('complex9')

    # Initialize and run DStream
    # denstream = DStream(lambda_=0.0001, epsilon=0.5, beta=0.3, mu=5)
    # denstream.fit(X)
    # labels = denstream.predict()

    # Evaluate
    # metrics = evaluate_clustering(X, labels)
    # print(metrics)

    pass


if __name__ == '__main__':
    main()

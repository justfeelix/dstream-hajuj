"""
Main entry point for DStream experiments.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.dstream import DStream
from baseframework.datahandler import load_data

def main():
    print("Loading data...")
    X, _ = load_data('complex9')
    
    dstream = DStream(grid_size=0.05, decay_factor=0.99)
    
    batch_size = 1000
    num_batches = len(X) // batch_size
    
    print(f"Starting stream processing ({num_batches} batches)...")
    
    for i in range(num_batches + 1):
        # 1. Get the next chunk of data  -> simulate the stream
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X))
        batch = X[start_idx:end_idx]
        
        if len(batch) == 0:
            break
            
        # 2. Update the online summaries
        dstream.update(batch)
        
        # 3. Periodically run the offline phase 
        if i % 5 == 0: # Print every 5 batches
            labels, centers = dstream.predict()
            num_clusters = len(np.unique(labels[labels != -1]))
            print(f"Batch {i}: Found {num_clusters} clusters from {len(centers)} active grid cells")

if __name__ == '__main__':
    main()
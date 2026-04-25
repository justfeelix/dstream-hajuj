"""
Main entry point for DStream experiments.
"""

import sys
sys.path.insert(0, '.')

import numpy as np
from src.dstream import DStream
from baseframework.datahandler import load_data
import argparse

def main():

    parser = argparse.ArgumentParser(description="Run DStream Clustering Simulation")
    
    # Define all the configurable parameters with default values
    parser.add_argument('--dataset', type=str, default='complex9', help='Name of the dataset (e.g., complex9, rbf3, har)')
    parser.add_argument('--grid_size', type=float, default=0.05, help='Size of the grid cells')
    parser.add_argument('--decay', type=float, default=0.99, help='Decay factor (lambda) for aging weights')
    parser.add_argument('--dense_th', type=float, default=1.5, help='Weight threshold to become DENSE')
    parser.add_argument('--sparse_th', type=float, default=0.5, help='Weight threshold to become SPARSE (noise)')
    parser.add_argument('--batch_size', type=int, default=1000, help='Number of instances to process per batch')
    
    args = parser.parse_args()


    print(f"Loading data '{args.dataset}'...")
    X, y = load_data(args.dataset)
    

    dstream = DStream(
        grid_size=args.grid_size, 
        decay_factor=args.decay,
        dense_threshold=args.dense_th,
        sparse_threshold=args.sparse_th
    )
    
    num_batches = len(X) // args.batch_size
    
    print(f"Starting stream processing ({num_batches} batches of size {args.batch_size})...")
    print(f"Parameters: grid={args.grid_size}, decay={args.decay}, dense_th={args.dense_th}, sparse_th={args.sparse_th}")
    
    for i in range(num_batches + 1):
        # 1. Get the next chunk of data  -> simulate the stream
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(X))
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
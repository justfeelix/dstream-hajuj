"""
Main entry point for DStream experiments.
"""

import sys
sys.path.insert(0, '.')

import argparse
import numpy as np
from src.dstream import DStream
from baseframework.datahandler import load_data
try:
    from baseframework.metrics.cmm import get_cmm
    from baseframework.metrics.TSindex import tempsil
except ImportError:
    print("Warning: Could not import metrics.")


def main():

    parser = argparse.ArgumentParser(description="Run DStream Clustering Simulation")
    
    # Define all the configurable parameters with default values
    parser.add_argument('--dataset', type=str, default='complex9', help='Name of the dataset (e.g., complex9, rbf3, har)')
    parser.add_argument('--grid_size', type=float, default=0.04, help='Size of the grid cells')
    parser.add_argument('--decay', type=float, default=0.99, help='Decay factor (lambda) for aging weights')
    parser.add_argument('--dense_th', type=float, default=1.5, help='Weight threshold to become DENSE')
    parser.add_argument('--sparse_th', type=float, default=0.5, help='Weight threshold to become SPARSE (noise)')
    parser.add_argument('--batch_size', type=int, default=500, help='Number of instances to process per batch')
    
    args = parser.parse_args()


    print(f"Loading data '{args.dataset}'...")
    X, y = load_data(args.dataset)
    

    dstream = DStream(
        grid_size=args.grid_size, 
        decay_factor=args.decay,
        dense_threshold=args.dense_th,
        sparse_threshold=args.sparse_th,
        cleanup_interval=args.batch_size 
    )
    
    num_batches = len(X) // args.batch_size
    
    print(f"\n=======================================================")
    print(f"STARTING STREAM PROCESSING: {num_batches} batches of size {args.batch_size}")
    print(f"PARAMETERS USED: grid={args.grid_size}, decay={args.decay}, dense_th={args.dense_th}, sparse_th={args.sparse_th}")
    print(f"=======================================================\n")
    
    for i in range(num_batches + 1):
        # 1. Get the next chunk of data  -> simulate the stream
        start_idx = i * args.batch_size
        end_idx = min((i + 1) * args.batch_size, len(X))
        batch = X[start_idx:end_idx]
        
        if len(batch) == 0:
            break
        # timestamp for this batch
        timestamps = np.arange(start_idx, end_idx)

        # 2. ONLINE PHASE (Learn)
        results = dstream.learn(batch, timestamps)

        logs = [res[2] for res in results if res[2] != ""]
        if logs:
            print(f"Batch {i} Event Logs: {len(logs)} state changes/cleanups occurred.")

        # 3. OFFLINE PHASE (Predict Assignment alternately)
        point_labels = dstream.predict(X_batch=batch)
        
        num_clusters = len(np.unique(point_labels[point_labels != -1]))
        print(f"   [Result] Batch {i} finished: {num_clusters} clusters found.")

        
    print("\n=======================================================")
    print("FINAL EVALUATION ON ENTIRE DATASET")
    print("=======================================================")
    
    # Map all points in X to the final clustered grid cells
    final_y_pred = dstream.predict(X_batch=X)
    
    print("\nCalculating Metrics...")
    try:
        # CMM: get_cmm(x, gt, clu, w, k)

        weights = np.ones(len(X)) # Give every point a weight of 1
        k_neighbors = 10          # KNN parameter for CMM calculation
        
        cmm_score = get_cmm(X, y, final_y_pred, weights, k_neighbors)
        print(f"CMM Score: {cmm_score:.4f}")
    except Exception as e:
        print(f"Could not calculate CMM: {e}")
        
    try:
        timestamps = np.arange(len(X))
        _, _, ts_score = tempsil(timestamps, X, final_y_pred)
        print(f"TS Index Score: {ts_score:.4f}")
    except Exception as e:
        print(f"Could not calculate TS Index: {e}")

if __name__ == '__main__':
    main()
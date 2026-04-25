"""
Online phase of DStream algorithm
"""


import numpy as np
from enum import Enum

class Status(Enum):
    DENSE = "dense"
    TRANSITIONAL = "normal"
    SPARSE = "noise"

class GridCell:
    def __init__(self, t_m):
        self.weight = 0.0 # initial weight
        self.t_m = t_m # last update time
        self.status = Status.NOISE # initial status


class OnlinePhase_StreamClusterer:
    def __init__(self, grid_size, decay_factor, dense_threshold, sparse_threshold):
        self.grid_size = grid_size
        self.decay_factor = decay_factor # or lambda
        self.dense_threshold = dense_threshold
        self.sparse_threshold = sparse_threshold
        
        # Summary: Dictionary with Grid Cell Ids (tuples) as keys and GridCell objects as values
        self.summaries = {} 
        self.current_time = 0

    def process_batch(self, batch):
        """Processes a chunk of incoming stream data."""
        # 1. Map every point in the batch to a grid index
        # 2. For each point:
        #    a. if its grid index is not in self.summaries, add it
        #    b. apply decay to the existing weight based on (current_time - cell.t_m)
        #    c. add +1 to the weight
        #    d. update cell.t_m to current_time

        # 1. map every point in the batch to a grid index
        grid_indices = np.floor(batch / self.grid_size).astype(int)
        #2.
        for i, idx_array in enumerate(grid_indices):
            # compute the exact time this specific point arrived
            point_time = self.current_time + i

            grid_tuple = tuple(idx_array)
            # a. add to summaries if its not there
            if grid_tuple not in self.summaries:
                self.summaries[grid_tuple] = GridCell(t_m=point_time)

            cell = self.summaries[grid_tuple]
            # b + c
            # exponential decay: Weight_{new} = Weight_{old} * \lambda^{\Delta t} + 1
            # where $\Delta t$ is how much time has passed since that specific grid cell was last updated
            cell.weight = cell.weight * (self.decay_factor ** (point_time - cell.t_m)) + 1
            cell.t_m = point_time
        
            #d. update cell.t_m to exact point_time
            cell.t_m = self.current_time

        # update global clock after the batch is finished
        self.current_time += len(batch)

    def update_status(self):
        """Updates the status of all grid cells based on thresholds"""
        # loop through self.summaries
        # if weight > dense_threshold -> status = "dense"
        # If weight < sparse_threshold -> status = "sparse" (or noise)
        for cell in self.summaries.values():
            if cell.weight >= self.dense_threshold:
                cell.status = Status.DENSE
            elif cell.weight <= self.sparse_threshold:
                cell.status = Status.SPARSE
            else:
                cell.status = Status.TRANSITIONAL
        

    def get_summary(self):
        """Returns the current summarization state"""
        return self.summaries
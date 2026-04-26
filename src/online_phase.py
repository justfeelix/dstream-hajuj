"""
Online phase of DStream algorithm
"""


import numpy as np
from enum import Enum
from baseframework.stream_clustering import OnlinePhase_StreamClusterer

class Status(Enum):
    DENSE = "dense"
    TRANSITIONAL = "normal"
    SPARSE = "noise"

class GridCell:
    def __init__(self,t_g, t_m=0.0):
        self.weight = 0.0 # initial weight
        self.t_g = t_g # last update time
        self.t_m = t_m # the last time this grid was deleted (grace period tracker)
        self.status = Status.SPARSE # initial status
        self.marked_sporadic = False # two-step deletion process


class DStreamOnlinePhase(OnlinePhase_StreamClusterer):
    def __init__(self, grid_size, decay_factor, dense_threshold, sparse_threshold, cleanup_interval=10, beta=0.3):
        super().__init__()
        self.grid_size = grid_size
        self.decay_factor = decay_factor # or lambda
        self.dense_threshold = dense_threshold
        self.sparse_threshold = sparse_threshold
        self.cleanup_interval = cleanup_interval
        self.beta = beta # controls the grace period for sporadic grids (in the paper)
        
        # Summary: Dictionary with Grid Cell Ids (tuples) as keys and GridCell objects as values
        self.summaries = {} 
        # track {grid_tuple: time_deleted} to enforce grace periods
        self.deleted_grids_times = {} 

        self.last_inspection_time = 0
        self.time_step = 0

    def learn_one(self, x, t: int) -> tuple[tuple, tuple, str]:
        """Processes a single instance and returns (summary_id, cluster_id, log_string)."""
        grid_array = np.floor(x / self.grid_size).astype(int)
        grid_tuple = tuple(grid_array)
        event_log = []
        
        # 1. create cell if it doesn't exist
        if grid_tuple not in self.summaries:
            past_t_m = self.deleted_grids_times.get(grid_tuple, 0)
            self.summaries[grid_tuple] = GridCell(t_g=t, t_m=past_t_m)

        cell = self.summaries[grid_tuple]
        old_status = cell.status
        
        # 2. apply decay
        cell.weight = cell.weight * (self.decay_factor ** (t - cell.t_g)) + 1
        cell.t_g = t

        # 3. update Status
        if cell.weight >= self.dense_threshold:
            cell.status = Status.DENSE
        elif cell.weight <= self.sparse_threshold:
            cell.status = Status.SPARSE
        else:
            cell.status = Status.TRANSITIONAL

        # 4. log status changes
        if old_status != cell.status:
            event_log.append(f"Grid {grid_tuple} status: {old_status.value} -> {cell.status.value}")

        self.time_step += 1

        # 5. periodic cleanup (garbage Collector)
        if self.time_step % self.cleanup_interval == 0:
            deleted_count = self._periodic_cleanup(t)
            if deleted_count > 0:
                event_log.append(f"Cleanup triggered: {deleted_count} sparse grids deleted.")

        log_string = " | ".join(event_log)
        
        # returns (summary_assignment_id, cluster_id, event_log)
        return (grid_tuple, grid_tuple, log_string)


    def _periodic_cleanup(self, current_time) -> int:
        keys_to_delete = []
        
        for grid_tuple, cell in self.summaries.items():
            # calculate the density threshold function (pi) simplified for fixed thresholds instead of dynamic density threshholds
            time_since_update = current_time - cell.t_g
            pi = self.sparse_threshold * (1 - (self.decay_factor ** (time_since_update + 1)))
            
            condition_s1 = cell.weight < pi
            condition_s2 = current_time >= (1 + self.beta) * cell.t_m
            
            if cell.marked_sporadic:
                if cell.t_g <= self.last_inspection_time:
                    keys_to_delete.append(grid_tuple)
                else:
                    if condition_s1 and condition_s2:
                        cell.marked_sporadic = True 
                    else:
                        cell.marked_sporadic = False 
            else:
                if condition_s1 and condition_s2:
                    cell.marked_sporadic = True 

        # start deletions
        for k in keys_to_delete:
            self.deleted_grids_times[k] = current_time
            del self.summaries[k]

        print(f"   [Garbage Collector] Deleted {len(keys_to_delete)} dead grid cells from memory.")
        self.last_inspection_time = current_time
        return len(keys_to_delete)
    

    def get_summary(self) -> dict:
        """Returns the current summarization state"""
        return self.summaries
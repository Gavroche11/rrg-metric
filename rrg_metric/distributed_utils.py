import torch
import torch.distributed as dist
import os
import math
import pickle

def is_distributed():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    if not is_distributed():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def get_world_size():
    if not is_distributed():
        return 1
    return dist.get_world_size()

def check_distributed_init():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

def split_data(data, rank, world_size):
    """Splits a list of data into chunks for each process."""
    if world_size == 1:
        return data
    
    # Simple chunking
    chunk_size = math.ceil(len(data) / world_size)
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, len(data))
    
    return data[start_idx:end_idx]

def gather_results(local_results):
    """
    Gathers results from all processes to rank 0.
    
    Args:
        local_results: The result object from the current process. 
                       Must be picklable.
    
    Returns:
        List of gathered results on rank 0 (flattened if local_results is a list).
        None on other ranks.
    """
    if not is_distributed():
        return local_results

    world_size = get_world_size()
    rank = get_rank()

    # Gather all results
    gathered_results = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_results, local_results)

    if rank == 0:
        # If the results are lists, flatten them
        if isinstance(local_results, list):
            flattened = []
            for res in gathered_results:
                if res is not None:
                    flattened.extend(res)
            return flattened
        else:
            # If not lists, just return the list of results
            return gathered_results
    else:
        return None
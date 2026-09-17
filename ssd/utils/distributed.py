"""Process groups shared by SSD's target and draft workers."""
from datetime import timedelta

import torch.distributed as dist


def init_model_parallel(config, rank, num_tp_gpus, device, backend="nccl"):
    timeout = timedelta(seconds=config.distributed_timeout_seconds)
    kwargs = {"device_id": device} if backend == "nccl" else {}
    dist.init_process_group(backend, config.distributed_init_method,
                            world_size=config.num_gpus, rank=rank,
                            timeout=timeout, **kwargs)
    return dist.new_group(ranks=list(range(num_tp_gpus)), timeout=timeout)


def create_async_group(config):
    return dist.new_group(ranks=[0, config.num_gpus - 1],
                          timeout=timedelta(seconds=config.distributed_timeout_seconds))

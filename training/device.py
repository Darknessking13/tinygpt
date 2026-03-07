"""Hardware auto-detection: CPU / single GPU / multi-GPU DDP."""
import os
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Literal


@dataclass
class DeviceContext:
    device: torch.device
    rank: int
    local_rank: int
    world_size: int
    is_main: bool
    backend: Literal["cpu", "single_gpu", "ddp"]
    amp_dtype: torch.dtype
    use_scaler: bool


def setup_device() -> DeviceContext:
    n_gpus = torch.cuda.device_count()

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        if torch.cuda.is_bf16_supported():
            amp_dtype, use_scaler = torch.bfloat16, False
        else:
            amp_dtype, use_scaler = torch.float16, True
        return DeviceContext(device, rank, local_rank, world_size, rank == 0,
                           "ddp", amp_dtype, use_scaler)

    if n_gpus >= 1:
        device = torch.device("cuda:0")
        if torch.cuda.is_bf16_supported():
            amp_dtype, use_scaler = torch.bfloat16, False
        else:
            amp_dtype, use_scaler = torch.float16, True
        return DeviceContext(device, 0, 0, 1, True, "single_gpu", amp_dtype, use_scaler)

    return DeviceContext(torch.device("cpu"), 0, 0, 1, True, "cpu", torch.bfloat16, False)


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def log(ctx: DeviceContext, msg: str):
    if ctx.is_main:
        print(msg)


def wrap_ddp(model: torch.nn.Module, ctx: DeviceContext) -> torch.nn.Module:
    if ctx.backend == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        return DDP(model, device_ids=[ctx.local_rank])
    return model


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model

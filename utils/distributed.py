import torch
import torch.distributed as dist

def _gather_from_all(tensor):
    gathered_tensor = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensor, tensor)
    gathered_tensor = torch.cat(gathered_tensor, 0)
    return gathered_tensor
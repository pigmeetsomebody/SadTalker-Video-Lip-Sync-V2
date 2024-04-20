import torch
import torch.distributed as dist
import numpy as np


def reduce_tensors(metrics):
    new_metrics = {}
    for k, v in metrics.items():
        if isinstance(v, torch.Tensor):
            dist.all_reduce(v)
            v = v / dist.get_world_size()
        if type(v) is dict:
            v = reduce_tensors(v)
        new_metrics[k] = v
    return new_metrics


def tensors_to_scalars(tensors):
    if isinstance(tensors, torch.Tensor):
        tensors = tensors.item()
        return tensors
    elif isinstance(tensors, dict):
        new_tensors = {}
        for k, v in tensors.items():
            v = tensors_to_scalars(v)
            new_tensors[k] = v
        return new_tensors
    elif isinstance(tensors, list):
        return [tensors_to_scalars(v) for v in tensors]
    else:
        return tensors


def convert_to_np(tensors):
    if isinstance(tensors, np.ndarray):
        return tensors
    elif isinstance(tensors, dict):
        new_np = {}
        for k, v in tensors.items():
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = convert_to_np(v)
            new_np[k] = v
    elif isinstance(tensors, list):
        new_np = []
        for v in tensors:
            if isinstance(v, torch.Tensor):
                v = v.cpu().numpy()
            if type(v) is dict:
                v = convert_to_np(v)
            new_np.append(v)
    elif isinstance(tensors, torch.Tensor):
        v = tensors
        if isinstance(v, torch.Tensor):
            v = v.cpu().numpy()
        if type(v) is dict:
            v = convert_to_np(v)
        new_np = v
    else:
        raise Exception(f'tensors_to_np does not support type {type(tensors)}.')
    return new_np


def convert_to_tensor(arrays):
    if isinstance(arrays, np.ndarray):
        v = torch.from_numpy(arrays).float()
        ret = v
    elif isinstance(arrays, torch.Tensor):
        ret = arrays
    elif isinstance(arrays, list):
        v = torch.from_numpy(np.array(arrays)).float()
    elif type(arrays) is dict:
        ret = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).float()
            if type(v) is dict:
                v = convert_to_tensor(v)
            ret[k] = v
    return ret

def convert_like(inp, target):
    if isinstance(target, np.ndarray):
        return convert_to_np(inp)
    elif isinstance(target, torch.Tensor):
        inp = convert_to_tensor(inp)
        inp = inp.to()
        if target.device == 'cpu':
            return move_to_cpu(inp)
        else:
            return move_to_cuda(inp)

def move_to_cpu(tensors):
    ret = {}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            v = v.cpu()
        if type(v) is dict:
            v = move_to_cpu(v)
        ret[k] = v
    return ret


def move_to_cuda(batch, gpu_id=0):
    # base case: object can be directly moved using `cuda` or `to`
    if callable(getattr(batch, 'cuda', None)):
        return batch.cuda(gpu_id, non_blocking=True)
    elif callable(getattr(batch, 'to', None)):
        return batch.to(torch.device('cuda', gpu_id), non_blocking=True)
    elif isinstance(batch, list):
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return batch
    elif isinstance(batch, tuple):
        batch = list(batch)
        for i, x in enumerate(batch):
            batch[i] = move_to_cuda(x, gpu_id)
        return tuple(batch)
    elif isinstance(batch, dict):
        for k, v in batch.items():
            batch[k] = move_to_cuda(v, gpu_id)
        return batch
    elif isinstance(batch, int) or isinstance(batch, float) or isinstance(batch, str):
        return batch
    elif batch is None:
        return None
    else:
        print("| Error in move_to_batch: ",type(batch), batch)
        raise NotImplementedError()
    return batch

def convert_to_half(arrays):
    if isinstance(arrays, np.ndarray):
        v = torch.from_numpy(arrays).half()
        ret = v
    elif isinstance(arrays, torch.Tensor):
        ret = arrays.half()
    elif isinstance(arrays, list):
        ret = [None for _ in range(len(arrays))]
        for i, v in enumerate(arrays):
            ret[i] = v.half()
    elif type(arrays) is dict:
        ret = {}
        for k, v in arrays.items():
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).half()
            if type(v) is dict:
                v = convert_to_tensor(v)
            ret[k] = v
    return ret
"""跑在 vLLM worker 进程里的函数 (经 ``collective_rpc`` 下发, 每个 TP worker 各跑一遍)。

第一个参数是 vLLM 的 worker 对象。要能 pickle 过去 (sampling.py 已经打开了
VLLM_ALLOW_INSECURE_SERIALIZATION), 所以都是模块级函数。
"""

from __future__ import annotations


def device_uuid(worker) -> str:
    """这个 worker 所在卡的 UUID。训练侧按它把 IPC handle 配给同卡的 worker。"""
    import torch

    return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)


def kv_sleep(worker) -> tuple[str, int]:
    """只 unmap tag=kv_cache 的 cumem 分配 (KV + 混合模型的状态缓存), 权重不动。

    不用 vLLM 自带的 sleep: 它连权重一起 unmap/remap, PPU 上之后重放 CUDA graph 段错误
    (见 arl/colocate/__init__.py)。调用前必须已经暂停调度 (controller 先 pause_generation),
    否则在跑的请求会读到被拿走的显存。返回 (卡 UUID, 放掉的字节数), 唤醒前训练侧按它查
    显存够不够。
    """
    import gc

    import torch
    from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release

    alloc = CuMemAllocator.get_instance()
    freed = 0
    for data in alloc.pointer_to_data.values():
        if data.tag == "kv_cache":
            unmap_and_release(data.handle)
            freed += data.handle[1]
    gc.collect()
    torch.cuda.empty_cache()
    return device_uuid(worker), freed


def kv_wake(worker) -> bool:
    """把 kv_sleep 放掉的分配按原虚拟地址重新映射 (内容是新的空显存)。"""
    from vllm.device_allocator.cumem import CuMemAllocator

    CuMemAllocator.get_instance().wake_up(tags=["kv_cache"])
    return True


def ipc_load_weights(worker, handles_by_uuid: dict) -> int:
    """``handles_by_uuid[uuid] = [(name, (rebuild_fn, args)), ...]``: 挑自己这张卡的那份,
    重建成张量 (和训练侧共享同一块显存, 不拷贝) 交给 ``load_weights`` (原地 copy_, 它自己按
    TP 切片)。返回载入的张量数。

    名字是训练侧 HF 文本类的名字; 多模态包装模型要折回 checkpoint 格式, 同
    ``sampling._nccl_update_weights_worker``。
    """
    import torch

    dev = torch.cuda.current_device()
    mine = handles_by_uuid.get(device_uuid(worker))
    if mine is None:
        raise RuntimeError(
            f"[colocate] no IPC handles for this worker's device ({device_uuid(worker)}); "
            f"got {list(handles_by_uuid)} -- the training rank on this card did not take part in the sync"
        )
    model = worker.model_runner.model
    remap_mm = hasattr(model, "language_model")
    weights = []
    for name, (fn, args) in mine:
        args = list(args)
        args[6] = dev  # storage_device: 发送方进程里的设备序号, 换成本进程的
        if remap_mm and name.startswith("model."):
            name = "model.language_model." + name[len("model."):]
        weights.append((name, fn(*args)))
    model.load_weights(weights)
    torch.cuda.synchronize()
    n = len(weights)
    del weights
    torch.cuda.ipc_collect()
    return n

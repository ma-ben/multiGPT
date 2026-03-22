import os

import torch
import torch.distributed as dist

import picotron.process_group_manager as pgm


STEP, VERBOSE = 0, os.environ.get("VERBOSE", "0") == "1"
_PIPELINE_COMM_STREAMS = {}
_PENDING_SEND_OPERATIONS = []


def _get_comm_stream(device):
    """
    为每个 CUDA 设备维护一个独立的 pipeline 通信流。

    这样做的目的不是“只要有 stream 就一定 overlap”，而是先把计算流和通信流分离开：
    1. 计算仍在默认流推进；
    2. P2P 通信尽量在专用流上排队；
    3. 需要真正消费结果时，再用 event 把依赖接回来。
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return None

    device_key = (device.type, device.index if device.index is not None else torch.cuda.current_device())
    if device_key not in _PIPELINE_COMM_STREAMS:
        _PIPELINE_COMM_STREAMS[device_key] = torch.cuda.Stream(device=device)
    return _PIPELINE_COMM_STREAMS[device_key]


def _enqueue_p2p_operations(ops, device, wait_for_completion, tensor_refs=None):
    """
    把一组 P2P 操作放到通信流上。

    `wait_for_completion=False` 只用于 send-only 场景。
    这样可以让发送在后台继续推进，而本 rank 先回到下一段本地计算。

    为了保证张量生命周期安全，异步 send 时会额外保存 `tensor_refs`。
    否则 Python 侧如果提前释放张量，底层通信仍可能在访问这块内存。
    """
    comm_stream = _get_comm_stream(device)

    if comm_stream is None:
        requests = dist.batch_isend_irecv(ops)
        if wait_for_completion:
            for request in requests:
                request.wait()
            return

        _PENDING_SEND_OPERATIONS.append(
            {
                "requests": requests,
                "event": None,
                "device": device,
                "tensor_refs": list(tensor_refs or []),
            }
        )
        return

    current_stream = torch.cuda.current_stream(device)
    producer_event = torch.cuda.Event()
    current_stream.record_event(producer_event)

    with torch.cuda.stream(comm_stream):
        # 通信流必须等到默认计算流上的 tensor 真正就绪，才能安全发起 send/recv。
        comm_stream.wait_event(producer_event)
        requests = dist.batch_isend_irecv(ops)

    completion_event = torch.cuda.Event()
    comm_stream.record_event(completion_event)

    if wait_for_completion:
        for request in requests:
            request.wait()
        # 当前计算流在使用 recv 到的 tensor 前，显式等待通信完成事件。
        current_stream.wait_event(completion_event)
        return

    _PENDING_SEND_OPERATIONS.append(
        {
            "requests": requests,
            "event": completion_event,
            "device": device,
            "tensor_refs": list(tensor_refs or []),
        }
    )


def drain_pipeline_communications():
    """
    等待所有延迟完成的 send-only 通信。

    这一步通常放在一个 pipeline step 的末尾：
    - 训练过程允许 send 在 step 内与部分本地计算重叠；
    - optimizer.step 前再统一收口，保证没有悬空通信留到下一个 step。
    """
    while _PENDING_SEND_OPERATIONS:
        pending = _PENDING_SEND_OPERATIONS.pop(0)
        for request in pending["requests"]:
            request.wait()

        if pending["event"] is not None and pending["device"].type == "cuda" and torch.cuda.is_available():
            torch.cuda.current_stream(pending["device"]).wait_event(pending["event"])


def pipeline_communicate(operation, device, dtype, tensor=None, shapes=None):
    global STEP
    global VERBOSE

    if operation == "recv_forward":
        if pgm.process_group_manager.pp_is_first_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_prev_rank
        is_send = False
        peer_rank = src
    elif operation == "send_forward":
        if pgm.process_group_manager.pp_is_last_stage:
            return
        dest = pgm.process_group_manager.pp_next_rank
        is_send = True
        peer_rank = dest
    elif operation == "recv_backward":
        if pgm.process_group_manager.pp_is_last_stage:
            return None
        tensor = torch.empty(shapes, requires_grad=True, device=device, dtype=dtype)
        src = pgm.process_group_manager.pp_next_rank
        is_send = False
        peer_rank = src
    elif operation == "send_backward":
        if pgm.process_group_manager.pp_is_first_stage:
            return
        dest = pgm.process_group_manager.pp_prev_rank
        is_send = True
        peer_rank = dest
    else:
        raise ValueError(f"Unsupported pipeline communication operation: {operation}")

    op = dist.P2POp(dist.isend if is_send else dist.irecv, tensor, peer_rank)
    if VERBOSE:
        print(
            f"{operation} | {'sending' if is_send else 'receiving'} {operation.split('_')[1]} "
            f"{pgm.process_group_manager.pp_rank} {'→' if is_send else '←'} {peer_rank} | "
            f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}",
            flush=True,
        )

    # send-only 可以后台推进；recv 必须等待，因为后续计算立刻依赖结果。
    _enqueue_p2p_operations(
        ops=[op],
        device=device,
        wait_for_completion=not is_send,
        tensor_refs=[tensor] if is_send and tensor is not None else None,
    )

    if VERBOSE:
        STEP += 1
    return tensor if not is_send else None


def bidirectional_pipeline_communicate(operation, send_tensor, recv_shapes, device, dtype):
    global STEP
    global VERBOSE

    is_fwd = operation == "send_fwd_recv_bwd"
    if (is_fwd and pgm.process_group_manager.pp_is_last_stage) or (not is_fwd and pgm.process_group_manager.pp_is_first_stage):
        return None

    peer_rank = pgm.process_group_manager.pp_next_rank if is_fwd else pgm.process_group_manager.pp_prev_rank
    recv_tensor = torch.empty(recv_shapes, requires_grad=True, device=device, dtype=dtype)

    if VERBOSE:
        print(
            f"{operation} | sending {'next' if is_fwd else 'prev'} {pgm.process_group_manager.pp_rank} -> {peer_rank} | "
            f"receiving {'next' if is_fwd else 'prev'} {peer_rank} -> {pgm.process_group_manager.pp_rank} | "
            f"STEP:{STEP} | RANK:{pgm.process_group_manager.pp_rank}",
            flush=True,
        )

    # 双向收发里 recv 结果会立刻用于后续计算，因此这里必须同步等待。
    _enqueue_p2p_operations(
        ops=[
            dist.P2POp(dist.isend, send_tensor, peer_rank),
            dist.P2POp(dist.irecv, recv_tensor, peer_rank),
        ],
        device=device,
        wait_for_completion=True,
        tensor_refs=[send_tensor],
    )

    if VERBOSE:
        STEP += 1
    return recv_tensor

import os
import time
import contextlib

import torch
from torch.profiler import ProfilerActivity, profile, record_function, schedule, tensorboard_trace_handler


class StepTimer:
    """
    轻量级 step 计时器。

    设计目标不是替代 profiler，而是给训练日志补一个“人能直接读懂”的阶段时间线。
    这样在还没打开 Chrome trace 之前，就能快速回答：
    1. 一个 step 主要花在了哪里；
    2. 是算子计算慢，还是同步/优化器阶段慢；
    3. 改完某个点以后，宏观时间有没有变化。
    """

    def __init__(self):
        self._durations = {}

    @contextlib.contextmanager
    def track(self, name: str):
        """
        同时做两件事：
        1. 用高精度时钟记录 wall-clock 时间；
        2. 用 record_function 给 profiler 打上可检索的阶段标签。

        这里故意保持实现极简，避免为了“监控”引入过重的运行时代码。
        """
        start_time = time.perf_counter()
        with record_function(name):
            yield
        self._durations[name] = self._durations.get(name, 0.0) + (time.perf_counter() - start_time)

    def format_summary(self) -> str:
        """
        统一输出毫秒级摘要，便于直接拼进训练日志。
        """
        if not self._durations:
            return "n/a"
        return " | ".join(f"{name}={duration * 1000.0:6.2f}ms" for name, duration in self._durations.items())


@contextlib.contextmanager
def profiling_context(config, is_profile_rank: bool):
    """
    条件式 profiler。

    只在指定 rank 上打开 profiler，原因有两个：
    1. profiler 本身有额外开销，不适合所有 rank 同时开；
    2. 多卡同时落 trace 会制造大量文件，先把“能读一张关键 rank 的时间线”做好更重要。
    """
    profiling_cfg = config.get("profiling") if hasattr(config, "get") else None
    profiler_enabled = bool(profiling_cfg.get("enabled", False)) if profiling_cfg is not None else False
    if not profiler_enabled or not is_profile_rank:
        yield None
        return

    output_dir = profiling_cfg.get("output_dir", "tmp/profiler")
    os.makedirs(output_dir, exist_ok=True)

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        schedule=schedule(
            wait=int(profiling_cfg.get("wait", 1)),
            warmup=int(profiling_cfg.get("warmup", 1)),
            active=int(profiling_cfg.get("active", 2)),
            repeat=int(profiling_cfg.get("repeat", 1)),
        ),
        on_trace_ready=tensorboard_trace_handler(output_dir),
        record_shapes=bool(profiling_cfg.get("record_shapes", True)),
        profile_memory=bool(profiling_cfg.get("profile_memory", True)),
        with_stack=bool(profiling_cfg.get("with_stack", False)),
    ) as profiler:
        yield profiler

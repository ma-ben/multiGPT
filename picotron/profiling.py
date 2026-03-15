import os
from contextlib import contextmanager, nullcontext

import torch
from torch.profiler import record_function


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "0").strip().lower()
    return value in {"1", "true", "yes", "on"}


@contextmanager
def profile_range(name: str):
    """Optional range annotation for torch profiler and Nsight Systems."""
    enable_record_function = _env_enabled("MULTIGPT_ENABLE_PROFILE_RANGES")
    enable_nvtx = _env_enabled("MULTIGPT_ENABLE_NVTX") and torch.cuda.is_available()

    record_ctx = record_function(name) if enable_record_function else nullcontext()
    with record_ctx:
        if enable_nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if enable_nvtx:
                torch.cuda.nvtx.range_pop()

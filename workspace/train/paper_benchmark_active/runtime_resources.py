from __future__ import annotations

import os
import random

import numpy as np
import torch


def default_cpu_threads() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, int(cpu_count) // 2)


def configure_runtime_resources(cpu_threads: int | None = None) -> int:
    threads = default_cpu_threads() if cpu_threads is None or int(cpu_threads) <= 0 else max(1, int(cpu_threads))
    for env_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(env_name, str(threads))

    effective_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", threads)))
    torch.set_num_threads(effective_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    return effective_threads


def configure_deterministic_research(seed: int = 0, *, deterministic: bool = True) -> int:
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return seed

import json
import os
import random
from typing import Any, Dict, Optional

import torch


def setup_reproducibility(
    deterministic: bool = False,
    disable_tf32: bool = False,
    matmul_precision: str = "highest",
    seed: int = 42,
    out_dir: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Optional reproducibility setup.

    Defaults are no-op for determinism/TF32. Seed is always set to
    stabilize any accidental randomness.
    """
    status: Dict[str, Any] = {}

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    if disable_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        except Exception as exc:
            status["disable_tf32_error"] = str(exc)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
            status["deterministic_algorithms"] = True
        except Exception as exc:
            status["deterministic_algorithms_error"] = str(exc)

    try:
        torch.set_float32_matmul_precision(matmul_precision)
    except Exception as exc:
        status["matmul_precision_error"] = str(exc)

    try:
        import numpy as np
    except Exception:
        np = None

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    status.update(
        {
            "deterministic": bool(deterministic),
            "disable_tf32": bool(disable_tf32),
            "matmul_precision": matmul_precision,
            "seed": int(seed),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "cuda_matmul_allow_tf32": getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            "cudnn_allow_tf32": getattr(torch.backends.cudnn, "allow_tf32", None),
            "cudnn_deterministic": getattr(torch.backends.cudnn, "deterministic", None),
            "cudnn_benchmark": getattr(torch.backends.cudnn, "benchmark", None),
        }
    )

    try:
        status["torch_float32_matmul_precision"] = torch.get_float32_matmul_precision()
    except Exception:
        status["torch_float32_matmul_precision"] = None

    if verbose:
        print("[Repro] settings:")
        for k, v in status.items():
            print(f"  {k}={v}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "repro.json"), "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    return status

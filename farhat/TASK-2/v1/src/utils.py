"""Utilitas bersama: logging, seed, IO json/npz, timing."""
from __future__ import annotations

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:  # pragma: no cover - torch opsional untuk modul ringan
        pass


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Timer:
    """Context manager sederhana untuk mencatat durasi tahap."""

    def __init__(self, label: str, logger: logging.Logger | None = None):
        self.label = label
        self.log = logger or get_logger("timer")

    def __enter__(self):
        self.t0 = time.time()
        self.log.info("start: %s", self.label)
        return self

    def __exit__(self, *exc):
        self.log.info("done : %s (%.1fs)", self.label, time.time() - self.t0)
        return False


def l2norm(x: np.ndarray, axis: int = 1, eps: float = 1e-8) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)

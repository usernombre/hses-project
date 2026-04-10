"""Data loading utilities for AES CPA datasets."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_TRACE_RE = re.compile(r"trace(\d+)\.txt$")

def load_cleartext(path: Path) -> np.ndarray:
    """Load cleartexts as a (num_traces, 16) uint8 array."""
    cleartext = np.loadtxt(path, dtype=np.uint8)
    if cleartext.ndim != 2 or cleartext.shape[1] != 16:
        raise ValueError(f"Expected cleartext shape (*, 16), got {cleartext.shape}")
    return cleartext

def trace_paths(dataset_dir: Path) -> list[Path]:
    """Return sorted trace file paths by byte index."""
    files = []
    for path in dataset_dir.glob("trace*.txt"):
        match = _TRACE_RE.search(path.name)
        if match:
            files.append((int(match.group(1)), path))
    if not files:
        raise FileNotFoundError(f"No traceN.txt files found in {dataset_dir}")
    files.sort(key=lambda item: item[0])
    return [path for _, path in files]

def load_trace(path: Path) -> np.ndarray:
    """Load one trace matrix as float32 with shape (num_traces, num_samples)."""
    trace = np.loadtxt(path, dtype=np.float32)
    if trace.ndim != 2:
        raise ValueError(f"Expected 2D trace matrix, got shape {trace.shape} from {path}")
    return trace

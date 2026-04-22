"""Data loading utilities for AES CPA datasets."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

def list_files(dataset_dir: Path, prefix: str) -> list[Path]:
    """Collect and sort indexed files such as traceN.txt or clockN.txt."""
    pattern = f"{prefix}*.txt"
    name_re = re.compile(rf"{re.escape(prefix)}(\d+)\.txt$")
    files = []
    for path in dataset_dir.glob(pattern):
        match = name_re.search(path.name)
        if match:
            files.append((int(match.group(1)), path))

    if not files:
        raise FileNotFoundError(f"Could not find {prefix} txt files")

    files.sort(key=lambda item: item[0])
    return [path for _, path in files]

def load_cleartext(path: Path) -> np.ndarray:
    """Load cleartexts as a (num_traces, 16) uint8 array."""
    cleartext = np.loadtxt(path, dtype=np.uint8)
    if cleartext.ndim != 2 or cleartext.shape[1] != 16:
        raise ValueError(f"Expected cleartext shape (*, 16), got {cleartext.shape}")
    return cleartext

def load_trace(path: Path) -> np.ndarray:
    """Load one trace matrix as float32 with shape (num_traces, num_samples)."""
    trace = np.loadtxt(path, dtype=np.float32)
    if trace.ndim != 2:
        raise ValueError(f"Expected 2D trace matrix, got shape {trace.shape} from {path}")
    return trace

def align_trace_with_clock(trace_matrix: np.ndarray, clock_matrix: np.ndarray) -> np.ndarray:
    """Align each trace row using piecewise warping between rising clock edges."""
    if trace_matrix.shape != clock_matrix.shape:
        raise ValueError(
            f"Trace and clock shapes differ: {trace_matrix.shape} vs {clock_matrix.shape}"
        )

    def rising_edges(row: np.ndarray) -> np.ndarray:
        threshold = 0.5 * (float(row.min()) + float(row.max()))
        logic = (row > threshold).astype(np.int8)
        return np.flatnonzero((logic[1:] - logic[:-1]) == 1).astype(np.int32)

    num_samples = trace_matrix.shape[1]
    sample_axis = np.arange(num_samples, dtype=np.float32)
    ref_edges = rising_edges(clock_matrix[0])
    aligned = np.zeros_like(trace_matrix)

    for idx in range(trace_matrix.shape[0]):
        row = trace_matrix[idx]
        cur_edges = rising_edges(clock_matrix[idx])

        if ref_edges.size < 2 or cur_edges.size < 2:
            aligned[idx] = row
            continue

        edge_count = int(min(ref_edges.size, cur_edges.size, 300))
        ref_pts = np.concatenate(([0], ref_edges[:edge_count], [num_samples - 1])).astype(np.float32)
        cur_pts = np.concatenate(([0], cur_edges[:edge_count], [num_samples - 1])).astype(np.float32)

        # Map each output sample to the source timeline so each clock edge lands on the reference edge.
        source_positions = np.interp(sample_axis, ref_pts, cur_pts)
        aligned[idx] = np.interp(source_positions, sample_axis, row).astype(trace_matrix.dtype)

    return aligned

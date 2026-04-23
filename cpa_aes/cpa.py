"""Correlation Power Analysis implementation for AES first-round attack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import align_trace_with_clock, list_files, load_cleartext, load_trace

AES_SBOX = np.array(
    [
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB,
        0x76, 0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4,
        0x72, 0xC0, 0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71,
        0xD8, 0x31, 0x15, 0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2,
        0xEB, 0x27, 0xB2, 0x75, 0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6,
        0xB3, 0x29, 0xE3, 0x2F, 0x84, 0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB,
        0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF, 0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45,
        0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8, 0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5,
        0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2, 0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44,
        0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73, 0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A,
        0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB, 0xE0, 0x32, 0x3A, 0x0A, 0x49,
        0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79, 0xE7, 0xC8, 0x37, 0x6D,
        0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08, 0xBA, 0x78, 0x25,
        0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A, 0x70, 0x3E,
        0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E, 0xE1,
        0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB,
        0x16,
    ],
    dtype=np.uint8,
)

HAMMING_WEIGHT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

@dataclass
class ByteResult:
    byte_index: int
    key_guess: int
    max_abs_correlation: float
    sample_index: int
    second_best_guess: int
    second_best_abs_correlation: float
    confidence_margin: float

def leakage_model(plaintext_byte_column: np.ndarray, key_guess: int) -> np.ndarray:
    """Predict Hamming-weight leakage for one byte guess after first-round AES SBox."""
    sbox_in = np.bitwise_xor(plaintext_byte_column, np.uint8(key_guess))
    sbox_out = AES_SBOX[sbox_in]
    return HAMMING_WEIGHT[sbox_out].astype(np.float32)

def pearson_against_trace_matrix(models: np.ndarray, trace_matrix: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation of one model vector against all trace samples."""
    models_centered = models - models.mean(axis=0, keepdims=True)
    trace_centered = trace_matrix - trace_matrix.mean(axis=0, keepdims=True)

    numerator = models_centered.T @ trace_centered
    model_energy = np.sum(models_centered * models_centered, axis=0)
    trace_energy = np.sum(trace_centered * trace_centered, axis=0)
    denom = np.sqrt(np.outer(model_energy, trace_energy))

    corr = None
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.divide(numerator, denom, out=np.zeros_like(numerator), where=denom > 0)
    return corr

def attack_one_byte(plaintext_column: np.ndarray, trace_matrix: np.ndarray, byte_index: int) -> ByteResult:
    """Run CPA for one key byte and return best and second-best guesses."""
    guesses = np.arange(256, dtype=np.uint8)
    sbox_in = np.bitwise_xor(plaintext_column[:, None], guesses[None, :])
    models = HAMMING_WEIGHT[AES_SBOX[sbox_in]].astype(np.float32)

    corr = pearson_against_trace_matrix(models, trace_matrix)
    abs_corr = np.abs(corr)
    best_samples = np.argmax(abs_corr, axis=1).astype(np.int32)
    best_scores = abs_corr[np.arange(256), best_samples].astype(np.float32)

    order = np.argsort(best_scores)[::-1]
    first = int(order[0])
    second = int(order[1])

    return ByteResult(
        byte_index=byte_index,
        key_guess=first,
        max_abs_correlation=float(best_scores[first]),
        sample_index=int(best_samples[first]),
        second_best_guess=second,
        second_best_abs_correlation=float(best_scores[second]),
        confidence_margin=float(best_scores[first] - best_scores[second]),
    )

def recover_key(dataset_dir: Path, clock_present: bool) -> tuple[np.ndarray, pd.DataFrame]:
    """Recover full 16-byte AES key from dataset trace files and cleartexts."""
    cleartext_path = dataset_dir / "cleartext.txt"
    cleartext = load_cleartext(cleartext_path)
    traces = list_files(dataset_dir, "trace")
    clocks = None
    if clock_present:
        print("Taking into account the clock for computations")
        clocks = list_files(dataset_dir, "clock")

    results: list[ByteResult] = []
    key = np.zeros(16, dtype=np.uint8)

    for byte_idx in range(16):
        print(f"Processing byte {byte_idx:2d}")
        trace_matrix = load_trace(traces[byte_idx])
        if clock_present:
            clock_matrix = load_trace(clocks[byte_idx])
            trace_matrix = align_trace_with_clock(trace_matrix, clock_matrix)
        result = attack_one_byte(cleartext[:, byte_idx], trace_matrix, byte_idx)
        results.append(result)
        key[byte_idx] = np.uint8(result.key_guess)
        print(
            f"    key=0x{result.key_guess:02x}  "
            f"confidence={result.max_abs_correlation:.4f}  "
            f"margin={result.confidence_margin:.4f}"
        )

    df = pd.DataFrame(
        {
            "byte_index": [r.byte_index for r in results],
            "key_guess_dec": [r.key_guess for r in results],
            "key_guess_hex": [f"0x{r.key_guess:02x}" for r in results],
            "max_abs_correlation": [r.max_abs_correlation for r in results],
            "sample_index": [r.sample_index for r in results],
            "second_best_guess_dec": [r.second_best_guess for r in results],
            "second_best_guess_hex": [f"0x{r.second_best_guess:02x}" for r in results],
            "second_best_abs_correlation": [r.second_best_abs_correlation for r in results],
            "confidence_margin": [r.confidence_margin for r in results],
        }
    )

    return key, df

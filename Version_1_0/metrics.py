import csv
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from skimage.metrics import structural_similarity as ssim_fn
# try:
#     from skimage.metrics import structural_similarity as ssim_fn
# except Exception:
#     ssim_fn = None

def _to_float01(frame: np.ndarray) -> np.ndarray:
    x = np.asarray(frame)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
    return np.clip(x, 0.0, 1.0)

def frame_ssim(frame_a: np.ndarray, frame_b: np.ndarray, channel_axis: Optional[int] = -1) -> float:
    # if ssim_fn is None:
    #     raise ImportError("scikit-image is required for SSIM: pip install scikit-image")

    a = _to_float01(frame_a)
    b = _to_float01(frame_b)

    if a.ndim == 2:
        return float(ssim_fn(a, b, data_range=1.0))
    return float(ssim_fn(a, b, data_range=1.0, channel_axis=channel_axis))

def frame_psnr(x: np.ndarray, y: np.ndarray):
    mse = np.mean((x - y) ** 2)
    if mse < 1e-12:
        return float("inf")
    return 10.0 * np.log10(255**2 / mse)


def evaluate_frame_pair_metrics(
    original_frame: np.ndarray,
    processed_frame: np.ndarray,
    # *,
    # compute_ssim: bool = True,
) -> Dict[str, float]:
    '''compute PNSR and SSIM for a pair of frames'''
    out = {
        "psnr": frame_psnr(original_frame, processed_frame),
        "ssim": frame_ssim(original_frame, processed_frame)
    }
    return out


def aggregate_frame_metrics(values: Sequence[float]) -> Dict[str, float]:
    vals = np.asarray(list(values), dtype=np.float64)
    if vals.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "median": float("nan"),
        }
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "median": float(np.median(vals)),
    }

def summarize_video_quality_metrics(frame_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Когда считать:
    - после завершения обработки всех кадров одного видео
    """
    if not frame_metrics:
        raise ValueError('frame_metrics is empty')

    keys = sorted(frame_metrics[0].keys())
    summary: Dict[str, float] = {}

    for k in keys:
        agg = aggregate_frame_metrics([fm[k] for fm in frame_metrics if k in fm])
        for stat_name, value in agg.items():
            summary[f"{k}_{stat_name}"] = value

    return summary


# =================================
# watermark_metrics
# =================================

def bit_error_rate(bits_true: Sequence[int], bits_pred: Sequence[int]) -> float:
    """
    Когда считать:
    - после извлечения watermark из clean video
    - после извлечения watermark из attacked video
    """
    a = np.asarray(bits_true, dtype=np.int64)
    b = np.asarray(bits_pred, dtype=np.int64)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in BER: {a.shape} vs {b.shape}")

    return float(np.mean(a != b))

# =================================
# safe saving
# =================================

class IncrementalCSVSink:
    """
    Пишет строки в CSV сразу после каждого видео.
    """

    def __init__(self, csv_path: str | Path, fieldnames: Sequence[str]):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)

        self._file = open(self.csv_path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fieldnames)

        if self.csv_path.stat().st_size == 0:
            self._writer.writeheader()
            self._flush()

    def _flush(self):
        self._file.flush()
        os.fsync(self._file.fileno())

    def write_row(self, row: Dict[str, Any]):
        safe_row = {k: row.get(k, None) for k in self.fieldnames}
        self._writer.writerow(safe_row)
        self._flush()

    def close(self):
        try:
            self._flush()
        finally:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
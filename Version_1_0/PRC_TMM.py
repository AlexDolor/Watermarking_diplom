import hashlib
import numpy as np
import cv2
import torch
import time
from tqdm import tqdm

from misc import encode_frame, decode_frame, detect_watermark_bit, make_progress_bar

def prc_generate_sequence(length: int, key: int) -> np.ndarray:
    """
    Генерирует длинную псевдослучайную бинарную последовательность {0,1}.
    """
    seed = int(hashlib.sha256(f"prc|{key}".encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = np.random.default_rng(seed)
    seq = rng.integers(low=0, high=2, size=length, dtype=np.int64)
    return seq.astype(np.uint8)


def prc_choose_start(video_name: str, key: int, max_start: int) -> int:
    """
    Детерминированно выбирает стартовый индекс для конкретного видео.
    """
    if max_start <= 0:
        return 0
    h = hashlib.sha256(f"start|{video_name}|{key}".encode("utf-8")).hexdigest()
    return int(h, 16) % max_start


def prc_get_video_bits(prc_sequence: np.ndarray, video_name: str, n_bits: int, key: int) -> tuple[np.ndarray, int]:
    """
    Возвращает фрагмент PRC-последовательности длины n_bits для данного видео.
    """
    if len(prc_sequence) < n_bits:
        raise ValueError("PRC sequence is shorter than requested n_bits")

    max_start = len(prc_sequence) - n_bits + 1
    start = prc_choose_start(video_name, key, max_start)
    bits = prc_sequence[start:start + n_bits].copy()
    return bits, start

# def prc_bit_for_frame(prc_sequence: np.ndarray, video_name: str, wm_local_idx: int, key: int) -> int:
#     """
#     wm_local_idx = локальный номер watermark-кадра внутри видео: 0,1,2,...
#     """
#     start = prc_choose_start(video_name, key, len(prc_sequence) - wm_local_idx)
#     return int(prc_sequence[start + wm_local_idx])

def edit_distance_bits(a: np.ndarray, b: np.ndarray):
    """
    Возвращает:
    - dist: edit distance
    a — целевая последовательность (например, встроенное сообщение m), длина n
    b — извлечённая последовательность, длина k.
    dp[i, j] будет хранить минимальное число операций (вставка, удаление, замена), 
        чтобы превратить первые i элементов a в первые j элементов b
    """
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)

    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return int(dp[n, m])

def random_match_pvalue(
    target_bits: np.ndarray,
    observed_bits: np.ndarray,
    n_trials: int = 1000,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    true_dist = edit_distance_bits(target_bits, observed_bits)

    random_dists = []
    for _ in range(n_trials):
        rand_bits = rng.integers(0, 2, size=len(target_bits), dtype=np.uint8)
        d = edit_distance_bits(rand_bits, observed_bits)
        random_dists.append(d)

    random_dists = np.array(random_dists, dtype=np.int32)
    p_value = float(np.mean(random_dists <= true_dist))

    return {
        "true_dist": int(true_dist),
        "random_mean_dist": float(random_dists.mean()),
        "random_std_dist": float(random_dists.std()),
        "p_value": p_value,
    }

# def tmm_find_best_match(prc_sequence: np.ndarray, observed_bits: np.ndarray, search_stride: int = 1):
#     """
#     Ищет лучшее место в длинной PRC-последовательности, куда можно выровнять observed_bits.

#     Возвращает:
#     - best_start
#     - best_dist
#     - best_ref_window
#     """
#     L = len(observed_bits)
#     if L == 0:
#         raise ValueError("observed_bits is empty")
#     if len(prc_sequence) < L:
#         raise ValueError("prc_sequence shorter than observed_bits")

#     best_start = None
#     best_dist = None
#     best_ref_window = None

#     for start in range(0, len(prc_sequence) - L + 1, search_stride):
#         ref_window = prc_sequence[start:start + L]
#         dist, _ = edit_distance_alignment(ref_window, observed_bits)

#         if best_dist is None or dist < best_dist:
#             best_dist = dist
#             best_start = start
#             best_ref_window = ref_window.copy()

#     return best_start, best_dist, best_ref_window

def decode_prc_sequence_from_video(
    input_path: str,
    vid_name: str,
    vae,
    opts,
):
    """
    Возвращает извлечённую последовательность битов по watermark-кадрам.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Empty video: {input_path}")

    # wm_frame_indices = set(
    #     np.linspace(
    #         0,
    #         max(frame_count - 1, 0),
    #         num=min(opts.wm_frames_per_video, frame_count),
    #         dtype=int,
    #     ).tolist()
    # )

    extracted_bits = []
    score_pairs = []

    frame_idx = 0
    pbar = make_progress_bar(total_frames, 'frames', 'Decoding sequence from video')

    while True:
        iter_start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break

        # if frame_idx not in wm_frame_indices:
        #     frame_idx += 1
        #     continue

        z = encode_frame(frame, vae, opts.device)
        # x = cv2_frame_to_vae_input(
        #     frame_bgr,
        #     device=opts.device,
        #     size=(opts.image_size, opts.image_size),
        # )

        # with torch.no_grad():
        #     posterior = vae.encode(x).latent_dist
        #     z = posterior.sample()
        #     z = z * vae.config.scaling_factor

        bit, score_pair = detect_watermark_bit(z, seed=opts.watermark_seed)
    
        extracted_bits.append(bit)
        score_pairs.append(score_pair)

        frame_idx += 1
        iter_time = time.perf_counter() - iter_start
        pbar.set_postfix_str(f"iter={iter_time:.3f}s")
        pbar.update(1)

    cap.release()

    return {
        "video_path": input_path,
        "video_name": vid_name,
        "bits": np.array(extracted_bits, dtype=np.uint8),
        "scores": score_pairs,
    }

def decode_video_with_prc_tmm(
    input_path: str,
    # vid_name: str,
    prc_sequence,
    vae,
    opts,
):
    '''Full decoding of the video'''
    if '\\' in input_path:
        vid_name = input_path.split('\\')[-1]
    else:
        vid_name = input_path.split('/')[-1]

    dec = decode_prc_sequence_from_video(
        input_path=input_path,
        vid_name=vid_name,
        vae=vae,
        opts=opts,
    )
    print('[INFO] Decoding prc sequence from video DONE')
    observed_bits = dec["bits"]

    target_bits, start = prc_get_video_bits(
        prc_sequence=prc_sequence,
        video_name=vid_name,
        n_bits=len(observed_bits),
        key=opts.watermark_seed,
    )

    print('[INFO] Random Match STARTED')
    stats = random_match_pvalue(
        target_bits=target_bits,
        observed_bits=observed_bits,
        n_trials=getattr(opts, "tmm_random_trials", 1000), #TODO
        seed=opts.watermark_seed,
    )
    print('[INFO] Random Match DONE')

    return {
        "video_path": input_path,
        "video_name": vid_name,
        "prc_start": int(start),
        "target_bits": target_bits,
        "observed_bits": observed_bits,
        "true_dist": stats["true_dist"],
        "random_mean_dist": stats["random_mean_dist"],
        "random_std_dist": stats["random_std_dist"],
        "p_value": stats["p_value"],
        "n_bits": int(len(observed_bits)),
    }
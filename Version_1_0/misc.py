import torch
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
import uuid

def encode_frame(frame: np.ndarray, vae, device: str):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(frame_rgb).float()          # [H, W, C]
    x = x / 255.0                                    # [0, 1]
    x = x.permute(2, 0, 1).unsqueeze(0)              # [1, C, H, W]
    x = x * 2.0 - 1.0                                # [-1, 1]
    x = x.to(device, dtype=torch.float32)
    return vae.encode(x)

def decode_frame(latent: torch.Tensor, vae):
    x = vae.decode(latent)
    x = x.detach().cpu().clamp(-1, 1)
    x = (x + 1.0) / 2.0                              # [0, 1]
    x = (x * 255.0).byte()[0]                        # [C, H, W]
    x = x.permute(1, 2, 0).numpy()                   # [H, W, C], RGB
    x_bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
    return x_bgr

def seeded_generator(seed: int, device: str):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g

def make_watermark_mask_like(z: torch.Tensor, seed: int, bit: int) -> torch.Tensor:
    """
    Возвращает маску {-1, +1} той же формы, что z.
    bit=1 -> mask
    bit=0 -> -mask
    """
    g = seeded_generator(seed, z.device)
    mask = torch.randint(
        low=0, high=2, size=z.shape, generator=g, device=z.device
    ).float()
    mask = mask * 2.0 - 1.0
    if bit == 0:
        mask = -mask
    return mask

def correlation_score(z: torch.Tensor, seed: int, bit: int) -> torch.Tensor:
    mask = make_watermark_mask_like(z, seed, bit)
    z_sign = torch.sign(z)
    score = (z_sign * mask).flatten(1).mean(dim=1)
    return score

def correlation_score_whitened(z: torch.Tensor, seed: int, bit: int, eps: float = 1e-8) -> torch.Tensor:
    mask = make_watermark_mask_like(z, seed, bit)

    zf = z.flatten(1)
    mf = mask.flatten(1)

    zf = zf - zf.mean(dim=1, keepdim=True)
    zf = zf / (zf.std(dim=1, keepdim=True) + eps)

    mf = mf - mf.mean(dim=1, keepdim=True)
    mf = mf / (mf.std(dim=1, keepdim=True) + eps)

    score = (zf * mf).mean(dim=1)
    return score

def detect_watermark_bit(z: torch.Tensor, seed: int):
    # s1 = correlation_score(z, seed, bit=1)
    # s0 = correlation_score(z, seed, bit=0)
    s1 = correlation_score_whitened(z, seed, bit=1)
    s0 = correlation_score_whitened(z, seed, bit=0)
    pred = (s1 > s0).long().cpu()
    return pred, (s1, s0)

def make_progress_bar(total_items: int, item_units: str, text_desc: str):
    '''make basic tqdm progress bar
    :params:
    total_items :int:
    item_units
    text_desc'''
    return tqdm(
        total=total_items,
        desc=text_desc,
        unit=item_units,
        dynamic_ncols=True,
        miniters=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

def save_side_by_side_comparison(original, watermarked, save_path):
    merged = cv2.hconcat([original, watermarked])
    cv2.imwrite(save_path, merged)

def make_unique_video_tag_strong(video_filename: str) -> str:
    stem = Path(video_filename).stem
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    short_uid = uuid.uuid4().hex[:8]
    return f"{stem}_{now_str}_{short_uid}"

def save_string_to_binary(value: str, out_path: str) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        f.write(value.encode("utf-8"))

def load_string_from_binary(in_path: str) -> str:
    with open(in_path, "rb") as f:
        return f.read().decode("utf-8")
import torch
import cv2
import numpy as np
from tqdm import tqdm

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


def detect_watermark_bit(z: torch.Tensor, seed: int):
    s1 = correlation_score(z, seed, bit=1)
    s0 = correlation_score(z, seed, bit=0)
    pred = (s1 > s0).long()
    return pred, (s1, s0)

def make_progress_bar(total_items: int, item_units: str, text_desc: str):
    return tqdm(
        total=total_items,
        desc=text_desc,
        unit=item_units,
        dynamic_ncols=True,
        miniters=1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )
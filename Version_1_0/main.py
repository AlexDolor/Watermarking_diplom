# ucf101_image_vae_watermark.py

import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List
import imageio.v2 as imageio

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import UCF101
from torchvision import transforms

# local imports
from dataset import UCF101FramesDataset
from AE import DiffusersVAEWrapper
from parser import print_opts, create_parser


# =========================
# 1. Конфиг
# =========================


# TODO переделать в парсер
@dataclass
class Config:
    root_dir: str = "./data/UCF101"
    annotation_path: str = "./data/ucfTrainTestlist"
    frames_per_clip: int = 16
    step_between_clips: int = 16
    frame_rate: Optional[int] = None

    image_size: int = 256
    batch_size: int = 4
    num_workers: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    watermark_seed: int = 1234
    watermark_strength: float = 0.15
    watermark_mode: str = "additive"   # "additive" or "sign_replace"
    watermark_bit: int = 1

    max_videos: Optional[int] = None
    output_dir: str = "./outputs"


CFG = Config()
os.makedirs(CFG.output_dir, exist_ok=True)





# =========================
# 3. Обёртка над UCF101
# =========================

# class UCF101FramesDataset(Dataset):
#     """
#     Достаёт из каждого клипа один центральный кадр.
#     Для baseline этого достаточно.
#     """
#     def __init__(self, cfg: Config, train: bool = True):
#         self.cfg = cfg
#         self.preprocess = FramePreprocess(cfg.image_size)

#         fold = 1
#         train_flag = train

#         self.dataset = UCF101(
#             root=cfg.root_dir,
#             annotation_path=cfg.annotation_path,
#             frames_per_clip=cfg.frames_per_clip,
#             step_between_clips=cfg.step_between_clips,
#             frame_rate=cfg.frame_rate,
#             fold=fold,
#             train=train_flag,
#             output_format="TCHW",
#         )

#         if cfg.max_videos is not None:
#             self.indices = list(range(min(cfg.max_videos, len(self.dataset))))
#         else:
#             self.indices = list(range(len(self.dataset)))

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         real_idx = self.indices[idx]
#         video, audio, label = self.dataset[real_idx]
#         # video: [T, C, H, W] при output_format="TCHW"
#         center = video.shape[0] // 2
#         frame = video[center]  # [C,H,W]
#         frame = self.preprocess(frame)
#         return {
#             "frame": frame,
#             "label": label,
#             "index": real_idx,
#         }





def load_pretrained_vae(device: str):
    """
    Здесь подключи свой реальный VAE.
    Например:
      - AutoencoderKL из diffusers
      - любой свой image AE/VAE

    Важно: вернуть объект с методами encode/decode.
    """
    # vae = DummyVAE().to(device).eval()
    vae = DiffusersVAEWrapper.to(device).eval()
    return vae


# =========================
# 5. Watermark-маска
# =========================

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


# =========================
# 6. Встраивание watermark
# =========================

def embed_watermark(
    z: torch.Tensor,
    seed: int,
    bit: int,
    strength: float = 0.15,
    mode: str = "additive",
    topk_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    mode="additive":
        z_wm = z + alpha * mask * |z|
    mode="sign_replace":
        z_wm = mask * |z|  только на top-k координатах

    topk_ratio: какая доля координат модифицируется
    """
    mask = make_watermark_mask_like(z, seed, bit)
    z_abs = z.abs()

    flat_abs = z_abs.flatten(1)
    k = max(1, int(flat_abs.shape[1] * topk_ratio))
    threshold = torch.topk(flat_abs, k=k, dim=1).values[:, -1].view(-1, 1, 1, 1)
    selector = (z_abs >= threshold).float()

    if mode == "additive":
        z_wm = z + strength * mask * z_abs * selector
    elif mode == "sign_replace":
        z_wm = z * (1.0 - selector) + (mask * z_abs) * selector
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return z_wm, selector


# =========================
# 7. Декодирование watermark
# =========================

def correlation_score(z: torch.Tensor, seed: int, bit: int) -> torch.Tensor:
    mask = make_watermark_mask_like(z, seed, bit)
    z_sign = torch.sign(z)
    score = (z_sign * mask).flatten(1).mean(dim=1)
    return score


def detect_watermark_bit(z: torch.Tensor, seed: int) -> torch.Tensor:
    s1 = correlation_score(z, seed, bit=1)
    s0 = correlation_score(z, seed, bit=0)
    pred = (s1 > s0).long()
    return pred


# =========================
# 8. Атаки для теста
# =========================

def jpeg_like_blur(x: torch.Tensor) -> torch.Tensor:
    """
    Очень грубая имитация потери качества.
    x в [-1,1]
    """
    x01 = (x + 1.0) / 2.0
    x_small = F.interpolate(x01, scale_factor=0.5, mode="bilinear", align_corners=False)
    x_back = F.interpolate(x_small, size=x01.shape[-2:], mode="bilinear", align_corners=False)
    x_back = x_back.clamp(0, 1)
    return x_back * 2.0 - 1.0


# =========================
# 9. Метрики
# =========================

def psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mse = F.mse_loss(x, y, reduction="none").flatten(1).mean(dim=1)
    return 10.0 * torch.log10(4.0 / (mse + 1e-8))  # т.к. диапазон [-1,1] => max^2 = 4

def save_watermarked_video(frames_tensor: torch.Tensor, save_path: str, fps: int = 8):
    """
    frames_tensor: [T, 3, H, W] в диапазоне [-1, 1]
    save_path: например "./outputs/sample_wm.mp4"
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    frames = ((frames_tensor.clamp(-1, 1) + 1.0) / 2.0 * 255.0).byte()
    frames = frames.permute(0, 2, 3, 1).cpu().numpy()  # [T, H, W, C]

    with imageio.get_writer(save_path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

# =========================
# 10. Основной цикл
# =========================

def run_baseline(opts):
    device = opts.device
    dataset = UCF101FramesDataset(
        data_dir=opts.data_dir,
        split=opts.split,
        image_size=opts.image_size,
        )
    loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        num_workers=opts.num_workers,
        pin_memory=True,
    )

    print('Dataset and loader done')

    vae = load_pretrained_vae(device)

    print('VAE loaded')

    all_psnr = []
    all_acc_clean = []
    all_acc_attack = []

    for batch in loader:
        x = batch["frame"].to(device)  # [B,3,H,W] in [-1,1]

        with torch.no_grad():
            z = vae.encode(x)
            z_wm, selector = embed_watermark(
                z,
                seed=opts.watermark_seed,
                bit=opts.watermark_bit,
                strength=opts.watermark_strength,
                mode=opts.watermark_mode,
                topk_ratio=0.10,
            )
            x_wm = vae.decode(z_wm).clamp(-1, 1)
            save_watermarked_video(x_wm, opts.output_dir)
            # качество
            batch_psnr = psnr(x, x_wm)
            all_psnr.extend(batch_psnr.detach().cpu().tolist())

            # clean detect
            z_clean_test = vae.encode(x_wm)
            pred_clean = detect_watermark_bit(z_clean_test, opts.watermark_seed)
            acc_clean = (pred_clean == opts.watermark_bit).float().mean()
            all_acc_clean.append(acc_clean.item())

            # attacked detect
            x_att = jpeg_like_blur(x_wm)
            z_att = vae.encode(x_att)
            pred_att = detect_watermark_bit(z_att, opts.watermark_seed)
            acc_att = (pred_att == opts.watermark_bit).float().mean()
            all_acc_attack.append(acc_att.item())
        break #TEMPORARY

    print(f"Mean PSNR:       {sum(all_psnr)/len(all_psnr):.3f}")
    print(f"Clean Bit Acc:   {sum(all_acc_clean)/len(all_acc_clean):.3f}")
    print(f"Attack Bit Acc:  {sum(all_acc_attack)/len(all_acc_attack):.3f}")


if __name__ == "__main__":
    parser = create_parser()
    opts, _ = parser.parse_known_args()
    print_opts(opts)
    run_baseline(opts)
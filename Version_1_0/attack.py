import io
import os
import math
import tempfile
import subprocess
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
import imageio.v2 as imageio

# ============================================================
# Helpers
# ============================================================

def _to_bcthw(video: torch.Tensor) -> torch.Tensor:
    if not isinstance(video, torch.Tensor):
        raise TypeError("video must be a torch.Tensor")
    if video.ndim == 4:
        return video.unsqueeze(0)
    elif video.ndim != 5:
        raise ValueError("video must have shape [B, C, T, H, W]")
    return video


def _flatten_bt(video: torch.Tensor) -> torch.Tensor:
    # [B, C, T, H, W] -> [C, B*T, H, W]
    b, c, t, h, w = video.shape
    # return video.permute(0, 2, 1, 3, 4).reshape(c, b * t, h, w)
    return video.permute(1, 0, 2, 3, 4).reshape(c, b * t, h, w)


def _unflatten_bt(frames: torch.Tensor, b: int, t: int) -> torch.Tensor:
    # [B*T, C, H, W] -> [B, C, T, H, W]
    _, c, h, w = frames.shape
    return frames.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def _odd(k: int) -> int:
    return k if k % 2 == 1 else k + 1


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype):
    kernel_size = _odd(kernel_size)
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel2d = torch.outer(g, g)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d


# ============================================================
# 1) Resize attack
# ============================================================

def attack_resize(
    video: torch.Tensor,
    scale: float,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Down/upscale each frame and return video back at the new size.
    If you want classic attack "downscale then back to original size",
    use attack_resize_roundtrip().
    """
    video = _to_bcthw(video)
    b, c, t, h, w = video.shape
    frames = _flatten_bt(video)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    resized = F.interpolate(
        frames, size=(new_h, new_w), mode=mode,
        align_corners=align_corners if mode in {"bilinear", "bicubic"} else None
    )
    return _unflatten_bt(resized, b, t)


def attack_resize_roundtrip(
    video: torch.Tensor,
    scale: float,
    down_mode: str = "bilinear",
    up_mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Classical resize attack:
    original -> scaled -> back to original resolution.
    """
    # video = _to_bcthw(video)
    # b, c, t, h, w = video.shape
    c, t, h, w = video.shape
    # frames = _flatten_bt(video)

    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    tmp = F.interpolate(
        video, size=(new_h, new_w), mode=down_mode,
        align_corners=align_corners if down_mode in {"bilinear", "bicubic"} else None
    )
    restored = F.interpolate(
        tmp, size=(h, w), mode=up_mode,
        align_corners=align_corners if up_mode in {"bilinear", "bicubic"} else None
    )
    return restored.clamp(0, 1)
    # return _unflatten_bt(restored, b, t).clamp(0, 1)


# ============================================================
# 2) Center crop + resize back
# ============================================================

def attack_center_crop_roundtrip(
    video: torch.Tensor,
    crop_ratio: float,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Crop central region and resize back to original HxW.
    crop_ratio in (0, 1], e.g. 0.9 means keep 90% of H and W.
    """
    video = _to_bcthw(video)
    b, c, t, h, w = video.shape
    frames = _flatten_bt(video)

    crop_h = max(1, int(round(h * crop_ratio)))
    crop_w = max(1, int(round(w * crop_ratio)))

    top = (h - crop_h) // 2
    left = (w - crop_w) // 2

    cropped = frames[:, :, top:top + crop_h, left:left + crop_w]
    restored = F.interpolate(
        cropped, size=(h, w), mode=mode,
        align_corners=align_corners if mode in {"bilinear", "bicubic"} else None
    )
    return _unflatten_bt(restored, b, t).clamp(0, 1)


# ============================================================
# 3) Random crop + resize back
# ============================================================

def attack_random_crop_roundtrip(
    video: torch.Tensor,
    crop_ratio: float,
    seed: Optional[int] = None,
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    """
    Random spatial crop, then resize back.
    """
    video = _to_bcthw(video)
    b, c, t, h, w = video.shape
    frames = _flatten_bt(video)

    crop_h = max(1, int(round(h * crop_ratio)))
    crop_w = max(1, int(round(w * crop_ratio)))

    g = torch.Generator(device=video.device)
    if seed is not None:
        g.manual_seed(seed)

    max_top = max(0, h - crop_h)
    max_left = max(0, w - crop_w)

    top = int(torch.randint(0, max_top + 1, (1,), generator=g, device=video.device).item())
    left = int(torch.randint(0, max_left + 1, (1,), generator=g, device=video.device).item())

    cropped = frames[:, :, top:top + crop_h, left:left + crop_w]
    restored = F.interpolate(
        cropped, size=(h, w), mode=mode,
        align_corners=align_corners if mode in {"bilinear", "bicubic"} else None
    )
    return _unflatten_bt(restored, b, t).clamp(0, 1)


# ============================================================
# 4) Brightness / contrast
# ============================================================

def attack_brightness(
    video: torch.Tensor,
    delta: float,
) -> torch.Tensor:
    """
    Add constant brightness shift.
    delta can be positive or negative, e.g. +0.05 or -0.05
    """
    video = _to_bcthw(video)
    return (video + delta).clamp(0, 1)


def attack_contrast(
    video: torch.Tensor,
    factor: float,
) -> torch.Tensor:
    """
    Contrast around per-frame mean.
    factor=1.0 means no change.
    """
    video = _to_bcthw(video)
    mean = video.mean(dim=(-1, -2), keepdim=True)
    out = (video - mean) * factor + mean
    return out.clamp(0, 1)


def attack_brightness_contrast(
    video: torch.Tensor,
    brightness_delta: float = 0.0,
    contrast_factor: float = 1.0,
) -> torch.Tensor:
    video = _to_bcthw(video)
    out = attack_contrast(video, factor=contrast_factor)
    out = attack_brightness(out, delta=brightness_delta)
    return out.clamp(0, 1)


# ============================================================
# 5) Additive Gaussian noise
# ============================================================

def attack_gaussian_noise(
    video: torch.Tensor,
    sigma: float,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Add iid Gaussian noise with std=sigma in [0,1] scale.
    Example sigma: 0.01, 0.02, 0.05
    """
    video = _to_bcthw(video)
    g = torch.Generator(device=video.device)
    if seed is not None:
        g.manual_seed(seed)

    noise = torch.randn(
        video.shape, generator=g, device=video.device, dtype=video.dtype
    ) * sigma
    return _flatten_bt((video + noise).clamp(0, 1))


# ============================================================
# 6) Gaussian blur
# ============================================================

def attack_gaussian_blur(
    video: torch.Tensor,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Spatial Gaussian blur frame-by-frame.
    """
    video = _to_bcthw(video)
    b, c, t, h, w = video.shape
    frames = _flatten_bt(video)

    kernel = _gaussian_kernel2d(kernel_size, sigma, device=video.device, dtype=video.dtype)
    kernel = kernel.view(1, 1, kernel.shape[0], kernel.shape[1]).repeat(c, 1, 1, 1)

    pad = kernel.shape[-1] // 2
    blurred = F.conv2d(frames, kernel, padding=pad, groups=c)
    return _unflatten_bt(blurred, b, t).clamp(0, 1)


# ============================================================
# 7) Salt-and-pepper-like impulse noise
# ============================================================

def attack_impulse_noise(
    video: torch.Tensor,
    prob: float = 0.01,
    salt_vs_pepper: float = 0.5,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Randomly set a fraction of pixels to 1 or 0.
    """
    video = _to_bcthw(video)
    g = torch.Generator(device=video.device)
    if seed is not None:
        g.manual_seed(seed)

    rnd = torch.rand(video.shape, generator=g, device=video.device)
    salt = torch.rand(video.shape, generator=g, device=video.device)

    out = video.clone()
    out[rnd < prob * salt_vs_pepper] = 1.0
    out[(rnd >= prob * salt_vs_pepper) & (rnd < prob)] = 0.0
    return out.clamp(0, 1)


# ============================================================
# 8) Temporal frame drop (simple)
# ============================================================

def attack_frame_drop(
    video: torch.Tensor,
    every_n: int = 2,
) -> torch.Tensor:
    """
    Drop every_n-th frame and restore original length
    by repeating nearest surviving frames.
    Output shape stays [B, C, T, H, W].
    Example: every_n=2 keeps frames [0,2,4,...] then repeats to T.
    """
    
    if every_n < 2:
        raise ValueError("every_n must be >= 2")

    video = _to_bcthw(video)
    b, c, t, h, w = video.shape

    if t == 0:
        raise ValueError("Temporal dimension T must be > 0")

    # keep all frames except every n-th one:
    # drop indices (every_n-1), (2*every_n-1), ...
    keep_mask = torch.ones(t, dtype=torch.bool, device=video.device)
    keep_mask[every_n - 1 :: every_n] = False

    keep_idx = torch.arange(t, device=video.device)[keep_mask]

    if keep_idx.numel() == 0:
        keep_idx = torch.tensor([0], device=video.device)

    kept = video[:, :, keep_idx, :, :]   # [B, C, T_keep, H, W]

    # restore back to original length T by nearest repetition
    restored_idx = torch.linspace(
        0, kept.shape[2] - 1, steps=t, device=video.device
    ).round().long()

    out = kept[:, :, restored_idx, :, :].contiguous()
    return _flatten_bt(out)


# ============================================================
# 9) ffmpeg H.264 compression attack
# ============================================================

def attack_h264_compression(
    video: torch.Tensor,
    fps: int = 24,
    crf: int = 28,
    preset: str = "medium",
    ffmpeg_path: str = "ffmpeg",
) -> torch.Tensor:
    """
    Non-differentiable attack through ffmpeg H.264 encode/decode.
    Requires imageio + imageio-ffmpeg OR ffmpeg installed in system PATH.
    Input/output: [B, C, T, H, W], values in [0,1].
    
    Applies compression independently per sample in batch.
    """
    # import imageio.v2 as imageio

    video = _to_bcthw(video)
    b, c, t, h, w = video.shape
    outs = []

    for i in range(b):
        sample = video[i].detach().cpu().clamp(0, 1)              # [C,T,H,W]
        sample = sample.permute(1, 2, 3, 0).mul(255).byte().numpy()  # [T,H,W,C]

        with tempfile.TemporaryDirectory() as tmpdir:
            inp = os.path.join(tmpdir, "inp.mp4")
            outp = os.path.join(tmpdir, "out.mp4")

            with imageio.get_writer(
                inp,
                fps=fps,
                codec="libx264",
                format="FFMPEG",
                pixelformat="yuv420p",
                ffmpeg_params=["-crf", str(crf), "-preset", preset],
            ) as writer:
                for frame in sample:
                    writer.append_data(frame)

            cmd = [
                ffmpeg_path, "-y", "-i", inp,
                "-c:v", "libx264",
                "-crf", str(crf),
                "-preset", preset,
                "-pix_fmt", "yuv420p",
                outp,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            reader = imageio.get_reader(outp, format="FFMPEG")
            frames = []
            for frame in reader:
                fr = torch.from_numpy(frame).float() / 255.0   # [H,W,C]
                frames.append(fr.permute(2, 0, 1))             # [C,H,W]
            reader.close()

        rec = torch.stack(frames, dim=0)                       # [T,C,H,W]

        # If decoder returned slightly different T, restore to original T
        if rec.shape[0] != t:
            rec_btchw = rec.unsqueeze(0).permute(0, 2, 1, 3, 4)   # [1,C,T,H,W]
            rec_btchw = F.interpolate(
                rec_btchw,
                size=(t, h, w),
                mode="trilinear",
                align_corners=False
            )
            rec = rec_btchw[0].permute(1, 0, 2, 3)  # [T,C,H,W]

        outs.append(rec.permute(1, 0, 2, 3))  # [C,T,H,W]
    
    out_vid = torch.stack(outs, dim=0).to(video.device, dtype=video.dtype).clamp(0, 1)
    # print(out_vid.shape)
    return _flatten_bt(out_vid)


def attack(input_path: str, output_path: str, device
    ) -> dict[str,str]:
    '''attack video from input path\n
    Implemented attacks:
    - resizing
    - frame dropping
    - H264 compression
    - gauss noise
    \n
    returns dict{Attack name : Attacked video path}'''
    video, fps = read_video_decord(input_path, device=device)

    resize_vid = attack_resize_roundtrip(video, 0.8)
    frame_drop_vid = attack_frame_drop(video, every_n=5)
    compressed_vid = attack_h264_compression(video, fps)
    gauss_noise_vid = attack_gaussian_noise(video, 0.02, 42)
    
    paths = {}
    paths['resized'] = output_path + '/resized.avi'
    paths['frame_dropped'] = output_path + '/frame_dropped.avi'
    paths['compressed'] = output_path + '/compressed.avi'
    paths['gauss_noised'] = output_path + '/gauss_noised.avi'

    save_video_imageio(resize_vid, paths['resized'], fps)
    save_video_imageio(frame_drop_vid, paths['frame_dropped'], fps)
    save_video_imageio(compressed_vid, paths['compressed'], fps)
    save_video_imageio(gauss_noise_vid, paths['gauss_noised'], fps)

    return paths


def read_video_decord(
    video_path: str,
    *,
    # target_size: Optional[Tuple[int, int]] = None,   # (H, W)
    # max_frames: Optional[int] = None,
    # sample_stride: int = 1,
    # output_format: str = "BCTHW",                    # "BCTHW" or "TCHW"
    device: Optional[str] = 'cpu',
):
    """
    Read video with decord.

    Returns:
        video:
            - [C, T, H, W]
          float32 in [0,1]
        fps: float
    """
    vr = VideoReader(video_path, ctx=cpu(0))
    fps = float(vr.get_avg_fps())

    # frame_indices = list(range(0, len(vr), sample_stride))
    # if max_frames is not None:
    #     frame_indices = frame_indices[:max_frames]

    # if len(frame_indices) == 0:
    #     raise ValueError(f"No frames selected from video: {path}")

    frames = vr.get_batch(list(range(0, len(vr)))).asnumpy()   # [T, H, W, C], uint8
    video = torch.from_numpy(frames).float() / 255.0
    # video = video.permute(0, 3, 1, 2).contiguous()   # [T, C, H, W]
    video = video.permute(3, 0, 1, 2).contiguous()  # [C, T, H, W]

    # if target_size is not None:
    #     th, tw = target_size
    #     video = F.interpolate(video, size=(th, tw), mode="bilinear", align_corners=False)

    # if device is not None:
    video = video.to(device)

    return video, fps

    # if output_format.upper() == "TCHW":
    #     return video, fps
    # elif output_format.upper() == "BCTHW":
    #     return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous(), fps
    # else:
    #     raise ValueError("output_format must be 'BCTHW' or 'TCHW'")


def save_video_imageio(video_tchw: torch.Tensor, path: str, fps: int = 24):
    '''
    video_tchw: [C, T, H, W], float in [0,1]
    ''' 
    # video_tchw: [C, T, H, W], float in [0,1]
    video = video_tchw.detach().cpu().clamp(0, 1)
    # video = (video * 255).byte().permute(0, 2, 3, 1).numpy()  # [T, H, W, C]
    video = (video * 255).byte().permute(1, 2, 3, 0).numpy()  # [T, H, W, C]

    print(path, '=='*30, video.shape)

    with imageio.get_writer(
        path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p"
    ) as writer:
        for frame in video:
            writer.append_data(frame)
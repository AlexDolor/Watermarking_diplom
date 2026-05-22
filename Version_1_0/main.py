import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, List
import imageio.v2 as imageio
import cv2
import subprocess
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision.datasets import UCF101
from torchvision import transforms

from dataset import UCF101FramesDataset, VideoPathDataset
from AE import DiffusersVAEWrapper
from parser import print_opts, create_parser
from PRC_TMM import prc_generate_sequence, prc_get_video_bits, decode_video_with_prc_tmm
from misc import encode_frame, decode_frame, make_watermark_mask_like, make_progress_bar, make_unique_video_tag_strong, save_string_to_binary
from attack import attack
from metrics import summarize_video_quality_metrics, evaluate_frame_pair_metrics, IncrementalCSVSink


def load_pretrained_vae(device: str):
    """
    Здесь подключи свой реальный VAE.
    Например:
      - AutoencoderKL из diffusers
      - любой свой image AE/VAE

    Важно: вернуть объект с методами encode/decode.
    """
    # vae = DummyVAE().to(device).eval()
    vae = DiffusersVAEWrapper(device=device).eval()
    return vae

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
        z_wm = z + strength * mask * |z|
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



def watermark_video(input_path, vid_name, vid_uuid, prc_sequence, vae, opts)-> Tuple:
    '''Full encoding cycle'''
    #Open video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {input_path}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames total ({total_frames/fps:.1f} seconds)")
    
    output_path = opts.output_dir + '/' + vid_name
    
    video_bits, prc_start = prc_get_video_bits(
		    prc_sequence=prc_sequence,
		    video_id=vid_uuid,
		    n_bits=total_frames,
		    key=opts.watermark_seed,
		)
    
    ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}', '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
            '-an',  # No audio for now
            output_path
            # opts.output_dir + '/' + vid_name
    ]
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL  # Prevent stderr buffer overflow causing deadlock
    )

    # Process frames
    frame_idx = 0

    psnr_watermarked = []
    frame_metrics = []
    pbar = make_progress_bar(total_frames, 'frames', 'Watermarking video')

    while True:
        iter_start = time.perf_counter()

        ret, frame = cap.read()
        if not ret:
            break
        
        # print(f'original img shape {frame.shape}')
        # print(f'original dtype {frame.dtype}')

        # Embed watermark
        z = encode_frame(frame, vae, opts.device)
        z_wm, selector = embed_watermark(
            z,
            seed=opts.watermark_seed,
            bit=int(video_bits[frame_idx]),
            strength=opts.watermark_strength,
            mode=opts.watermark_mode,
            topk_ratio=opts.topk_ratio
        )
        watermarked = decode_frame(z_wm, vae)
        # print(f'decoded img shape: {watermarked.shape}')
        # print(f'decoded dtype {watermarked.dtype}')

        #metrics
        # psnr_watermarked.append(np_psnr(watermarked, frame))
        frame_metrics.append(evaluate_frame_pair_metrics(frame, watermarked))
        # save_side_by_side_comparison( frame, watermarked, 
        #     opts.frame_comparison_dir + '/' + vid_name + f'_{watermarked_count}.jpg')

        # Write to FFmpeg
        proc.stdin.write(watermarked.tobytes())
    
        frame_idx += 1
        iter_time = time.perf_counter() - iter_start
        pbar.set_postfix_str(f"iter={iter_time:.3f}s")
        pbar.update(1)
        # if frame_idx % 10 == 0:
        #     progress = (frame_idx / total_frames) * 100
        #     print(f"[PROGRESS] Processed {frame_idx}/{total_frames} frames ({progress:.1f}%) - watermarked: {watermarked_count}", flush=True)

    cap.release()
    proc.stdin.close()
    proc.wait()

    metrics = summarize_video_quality_metrics(frame_metrics)
    return metrics, video_bits

# =========================
# 
# =========================

def main(opts):
    
    dataset = VideoPathDataset(
        data_dir=opts.data_dir,
        split=opts.data_split,
        # max_videos=opts.max_videos
        )
    
    if opts.max_videos == 0:
        opts.max_videos = len(dataset)
    else:   
        opts.max_videos = min(len(dataset), opts.max_videos)

    generator = torch.Generator()
    generator.manual_seed(420)
    sampler = RandomSampler(
        dataset,
        replacement=False,
        num_samples=opts.max_videos,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler
    )


    vae = load_pretrained_vae(opts.device)

    print('[INFO] datatset and VAE loaded successully')

    prc_sequence = prc_generate_sequence(
        length=opts.prc_length,
        key=opts.watermark_seed,
    )
    print('[INFO] prc_sequence generated')
    

    
    
    pbar = make_progress_bar(opts.max_videos, 'videos', 'Watermarking videos')

    # psnr_all = []
    fieldnames = [
        "video_id",
        'video_uuid',
        "attack_name",
        "psnr_mean",
        "ssim_mean",
        "edit_distance",
        "normalized_edit_distance",
        'p_value'
    ]
    with IncrementalCSVSink(opts.metrics_dir+"/metrics.csv", fieldnames) as sink:
        for sample in loader:
            iter_start = time.perf_counter()
            vid_uuid = make_unique_video_tag_strong('video_path')
            save_string_to_binary(vid_uuid, opts.output_dir + f'/uuid/{sample["file_name"][0]}')
            
            metrics, video_bits = watermark_video(
                input_path = sample['video_path'][0],
                vid_name = sample['file_name'][0],
                vid_uuid=vid_uuid,
                prc_sequence=prc_sequence,
                vae=vae, 
                opts=opts
                )
            # psnr_all.append(res[0])
            print(f'encoded video bits: {video_bits}')
            res = decode_video_with_prc_tmm(
                input_path=opts.output_dir + '/' + sample['file_name'][0],
                vid_id=vid_uuid,
                prc_sequence=prc_sequence,
                vae=vae,
                opts=opts
                )
            row = {
                "video_id": sample['file_name'][0],
                'video_uuid': vid_uuid,
                "attack_name": 'watermarking',
                "psnr_mean": metrics["psnr_mean"],
                "ssim_mean": metrics["ssim_mean"],
                "edit_distance": res["edit_distance"],
                "normalized_edit_distance": res["normalized_edit_distance"],
                'p_value': res['p_value']
            }

            sink.write_row(row)


            for item in res.items():
                print(f'{item[0]}:{item[1]}')

            paths = attack(opts.output_dir + '/' + sample['file_name'][0],
                        opts.output_dir, opts.device)
            
            for p in paths.items():
                res = decode_video_with_prc_tmm(
                    input_path=p[1],
                    vid_id=vid_uuid,
                    prc_sequence=prc_sequence,
                    vae=vae,
                    opts=opts
                    )
                row = {
                    "video_id": sample['file_name'][0],
                    'video_uuid': vid_uuid,
                    "attack_name": p[0],
                    "psnr_mean": None,
                    "ssim_mean": None,
                    "edit_distance": res["edit_distance"],
                    "normalized_edit_distance": res["normalized_edit_distance"],
                    'p_value': res['p_value']
                }

                sink.write_row(row)

                print('\n','='*80, '\n')
                for item in res.items():
                    print(f'{item[0]}:{item[1]}')

            iter_time = time.perf_counter() - iter_start
            pbar.set_postfix_str(f"iter={iter_time:.3f}s | file={sample['file_name'][0]}")
            pbar.update(1)

    
    #save metrics
    # pd.DataFrame(psnr_all, columns=['PSNR']).to_csv(opts.metrics_dir + '/psnr.csv')

    # print(f'[INFO] Watermark encoding and decoding done\nMean PSNR {np.mean(psnr_all)}')
    print(f'[INFO] Watermark encoding and decoding done')


if __name__ == "__main__":
    parser = create_parser()
    opts, _ = parser.parse_known_args()
    print_opts(opts)

    os.makedirs(opts.output_dir, exist_ok=True)
    os.makedirs(opts.frame_comparison_dir, exist_ok=True)
    os.makedirs(opts.metrics_dir, exist_ok=True)
    main(opts)
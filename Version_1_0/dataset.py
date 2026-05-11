from pathlib import Path
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset
# from torchvision.io import read_video DEPRECATED
from decord import VideoReader, cpu

from torchcodec.decoders import VideoDecoder
from torchvision import transforms

# =========================
# 2. Нормализация под VAE
# =========================

class FramePreprocess:
    """
    Преобразует кадр из uint8 [0,255] / float [0,1]
    в формат, удобный для image VAE:
    RGB, resize, float32, диапазон [-1, 1]
    """
    def __init__(self, image_size: int):
        self.image_size = image_size
        self.resize = transforms.Resize((image_size, image_size), antialias=True)

    def __call__(self, frame: torch.Tensor) -> torch.Tensor:
        # ожидаем H,W,C или C,H,W
        if frame.ndim == 3 and frame.shape[-1] == 3:
            frame = frame.permute(2, 0, 1)  # HWC -> CHW
        frame = frame.float()
        if frame.max() > 1.0:
            frame = frame / 255.0
        frame = self.resize(frame)
        frame = frame.clamp(0, 1)
        frame = frame * 2.0 - 1.0  # [0,1] -> [-1,1]
        return frame

class UCF101FramesDataset(Dataset):
    """
    Читает датасет в формате:

    root/
      train/
        ClassA/
          vid1.avi
          vid2.avi
      val/
        ClassB/
          vid3.avi
      test/
        ClassC/
          vid4.avi
      train.csv
      val.csv
      test.csv

    Для baseline:
    - берёт один центральный кадр из видео;
    - метку класса берёт из имени папки;
    - CSV не обязателен и по умолчанию не используется.
    """

    VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}

    def __init__(
        self,
        data_dir: str = './data/UCF101',
        split: str = "train",
        image_size: int = 256,
        max_videos: Optional[int] = None,
        preprocess=None,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.split_dir = self.data_dir / split
        self.image_size = image_size
        #TODO возможно переделать
        self.preprocess = preprocess if preprocess is not None else FramePreprocess(image_size)

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split folder not found: {self.split_dir}")

        self.class_names = sorted(
            [p.name for p in self.split_dir.iterdir() if p.is_dir()]
        )
        if len(self.class_names) == 0:
            raise RuntimeError(f"No class folders found in {self.split_dir}")

        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        self.samples: List[Dict] = []

        # if self.use_csv:
        #     csv_path = self.data_dir / f"{split}.csv"
        #     if not csv_path.exists():
        #         raise FileNotFoundError(f"CSV file not found: {csv_path}")
        #     self.samples = self._build_samples_from_csv(csv_path)
        # else:
        #     self.samples = self._build_samples_from_folders()
        self.samples = self._build_samples_from_folders()

        if max_videos is not None:
            self.samples = self.samples[:max_videos]

        if len(self.samples) == 0:
            raise RuntimeError(f"No video files found in {self.split_dir}")

    def _is_video_file(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.VIDEO_EXTS

    def _build_samples_from_folders(self) -> List[Dict]:
        samples = []
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            for video_path in sorted(class_dir.rglob("*")):
                if self._is_video_file(video_path):
                    samples.append(
                        {
                            "video_path": str(video_path),
                            "class_name": class_name,
                            "label": self.class_to_idx[class_name],
                        }
                    )
        return samples

    # def _build_samples_from_csv(self, csv_path: Path) -> List[Dict]:
    #     import pandas as pd

    #     df = pd.read_csv(csv_path)
    #     cols = [c.lower() for c in df.columns]

    #     possible_path_cols = ["filepath", "file_path", "path", "video", "video_path", "filename", "file"]
    #     possible_label_cols = ["label", "class", "class_name", "category"]

    #     path_col = None
    #     label_col = None

    #     for c in df.columns:
    #         if c.lower() in possible_path_cols:
    #             path_col = c
    #             break

    #     for c in df.columns:
    #         if c.lower() in possible_label_cols:
    #             label_col = c
    #             break

    #     if path_col is None:
    #         raise ValueError(
    #             f"Could not infer path column in {csv_path}. "
    #             f"Expected one of: {possible_path_cols}"
    #         )

    #     samples = []
    #     for _, row in df.iterrows():
    #         raw_path = str(row[path_col])

    #         video_path = Path(raw_path)
    #         if not video_path.is_absolute():
    #             candidate1 = self.data_dir / raw_path
    #             candidate2 = self.split_dir / raw_path

    #             if candidate1.exists():
    #                 video_path = candidate1
    #             elif candidate2.exists():
    #                 video_path = candidate2
    #             else:
    #                 video_path = candidate2

    #         if not video_path.exists():
    #             continue

    #         if label_col is not None:
    #             class_name = str(row[label_col])
    #         else:
    #             class_name = video_path.parent.name

    #         if class_name not in self.class_to_idx:
    #             # если в CSV метка отличается, но папка корректная — доверяем папке
    #             class_name = video_path.parent.name

    #         if class_name not in self.class_to_idx:
    #             continue

    #         samples.append(
    #             {
    #                 "video_path": str(video_path),
    #                 "class_name": class_name,
    #                 "label": self.class_to_idx[class_name],
    #             }
    #         )

    #     return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]

        vr = VideoReader(video_path, ctx=cpu(0))
        video = torch.from_numpy(vr[:].asnumpy())
        # video, _, info = read_video(video_path, pts_unit="sec")
        # video = VideoDecoder(video_path)
        # video: [T, H, W, C], uint8

        if video.shape[0] == 0:
            raise RuntimeError(f"Empty or unreadable video: {video_path}")

        center_idx = video.shape[0] // 2
        frame = video[center_idx]  # [H, W, C]

        frame = self.preprocess(frame)

        return {
            "frame": frame,                      # [C, H, W], float32, usually [-1,1]
            "label": sample["label"],           # int
            "class_name": sample["class_name"], # str
            "video_path": video_path,           # str
            "index": idx,
        }
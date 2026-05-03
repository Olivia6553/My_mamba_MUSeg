### 2026/4/27 wyj: 双分支改写


import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CenterBlockBranchDataset(Dataset):
    """
    从整图中只读取 (3,3) 和 (3,4) 两个中心块，
    并根据 7x8 ROI grid 过滤出 ROI 或 BG 样本。

    返回:
        block_tensor: [3,128,128]
        meta: dict
    """

    def __init__(
        self,
        image_dir: str,
        grid_dir: str,
        branch_type: str = "roi",   # "roi" or "bg"
        positions: List[Tuple[int, int]] = [(3, 3), (3, 4)],
        block_h: int = 128,
        block_w: int = 128,
    ):
        super().__init__()
        assert branch_type in ["roi", "bg"]

        self.image_dir = Path(image_dir)
        self.grid_dir = Path(grid_dir)
        self.branch_type = branch_type
        self.positions = positions
        self.block_h = block_h
        self.block_w = block_w

        self.samples = []

        image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        for img_path in image_paths:
            stem = img_path.stem

            # 兼容两种 grid 命名
            cand1 = self.grid_dir / f"{stem}.npy"
            cand2 = self.grid_dir / f"{stem}_label.npy"

            if cand1.exists():
                grid_path = cand1
            elif cand2.exists():
                grid_path = cand2
            else:
                continue

            grid = np.load(grid_path)   # [7,8]
            assert grid.shape == (7, 8), f"{grid_path} shape 不是 (7,8): {grid.shape}"

            for (r, c) in self.positions:
                label = int(grid[r, c])  # 0 or 1

                if self.branch_type == "roi" and label == 1:
                    self.samples.append((img_path, r, c, label))
                elif self.branch_type == "bg" and label == 0:
                    self.samples.append((img_path, r, c, label))

        if len(self.samples) == 0:
            raise ValueError(
                f"[{self.branch_type}] 没有找到样本，请检查 image_dir/grid_dir/ROI标签是否匹配。\n"
                f"image_dir={self.image_dir}\ngrid_dir={self.grid_dir}"
            )

        print(f"[CenterBlockBranchDataset] branch_type = {self.branch_type}")
        print(f"[CenterBlockBranchDataset] image_dir   = {self.image_dir}")
        print(f"[CenterBlockBranchDataset] grid_dir    = {self.grid_dir}")
        print(f"[CenterBlockBranchDataset] positions   = {self.positions}")
        print(f"[CenterBlockBranchDataset] 样本数       = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, r, c, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255.0   # [H,W,3]

        y0 = r * self.block_h
        y1 = (r + 1) * self.block_h
        x0 = c * self.block_w
        x1 = (c + 1) * self.block_w

        block = img[y0:y1, x0:x1, :]                     # [128,128,3]
        block = np.transpose(block, (2, 0, 1))          # [3,128,128]
        block = torch.from_numpy(block).float()

        meta = {
            "source_name": img_path.name,
            "row": r,
            "col": c,
            "label": label,
        }

        return block, meta
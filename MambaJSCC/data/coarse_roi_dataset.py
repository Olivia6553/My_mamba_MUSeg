### 2026/4/24：wyj 像素级分割数据

from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class CoarseROIPixelDataset(Dataset):
    """
    读取:
      - Image_1024x896/*.jpg
      - Label_1024x896/*.png

    返回:
      img: [3,896,1024]
      label: [896,1024]   原始语义标签图（像素级）
    """
    def __init__(self, image_dir: str, label_dir: str):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)

        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        self.samples = []
        for img_path in self.image_paths:
            stem = img_path.stem

            # 兼容 xxx.png 和 xxx_label.png
            label_path_1 = self.label_dir / f"{stem}.png"
            label_path_2 = self.label_dir / f"{stem}_label.png"

            if label_path_1.exists():
                label_path = label_path_1
            elif label_path_2.exists():
                label_path = label_path_2
            else:
                continue

            self.samples.append((img_path, label_path))

        if len(self.samples) == 0:
            raise ValueError("没有找到图像与像素标签图的匹配样本。")

        print(f"[CoarseROIPixelDataset] image_dir = {self.image_dir}")
        print(f"[CoarseROIPixelDataset] label_dir = {self.label_dir}")
        print(f"[CoarseROIPixelDataset] 样本数 = {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label_path = self.samples[idx]

        # 图像
        img = Image.open(img_path).convert("RGB")
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC -> CHW
        img = torch.from_numpy(img)

        # 像素级语义标签图
        label = np.array(Image.open(label_path), dtype=np.int64)
        label = torch.from_numpy(label)       # [896,1024]

        return img, label, img_path.stem

# ### 2026/4/21 wyj：粗ROI识别：搭建数据集读取代码

# from pathlib import Path
# import numpy as np
# from PIL import Image
# import torch
# from torch.utils.data import Dataset


# class CoarseROIDataset(Dataset):
#     """
#     读取:
#       - Image_1024x896/*.jpg
#       - BlockROI_binary_grid_7x8/*.npy

#     注意:
#       你当前 npy 里是 0 / 64
#       这里统一转成 0 / 1，方便训练
#     """
#     def __init__(self, image_dir: str, grid_dir: str):
#         self.image_dir = Path(image_dir)
#         self.grid_dir = Path(grid_dir)

#         self.image_paths = sorted([
#             p for p in self.image_dir.iterdir()
#             if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
#         ])

#         self.samples = []
#         for img_path in self.image_paths:
#             stem = img_path.stem

#             # 兼容 xxx.npy 和 xxx_label.npy
#             grid_path_1 = self.grid_dir / f"{stem}.npy"
#             grid_path_2 = self.grid_dir / f"{stem}_label.npy"

#             if grid_path_1.exists():
#                 grid_path = grid_path_1
#             elif grid_path_2.exists():
#                 grid_path = grid_path_2
#             else:
#                 continue

#             self.samples.append((img_path, grid_path))

#         if len(self.samples) == 0:
#             raise ValueError("没有找到图像与块标签的匹配样本，请检查文件名。")

#         print(f"[CoarseROIDataset] image_dir = {self.image_dir}")
#         print(f"[CoarseROIDataset] grid_dir  = {self.grid_dir}")
#         print(f"[CoarseROIDataset] 样本数 = {len(self.samples)}")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx: int):
#         img_path, grid_path = self.samples[idx]

#         # 图像 -> [3,896,1024], float32
#         img = Image.open(img_path).convert("RGB")
#         img = np.asarray(img, dtype=np.float32) / 255.0
#         img = np.transpose(img, (2, 0, 1))
#         img = torch.from_numpy(img)

#         # 标签 -> [1,7,8], float32, 0/1
#         grid = np.load(grid_path).astype(np.float32)
#         grid = (grid > 0).astype(np.float32)   # 0/64 -> 0/1
#         grid = torch.from_numpy(grid).unsqueeze(0)

#         return img, grid, img_path.stem
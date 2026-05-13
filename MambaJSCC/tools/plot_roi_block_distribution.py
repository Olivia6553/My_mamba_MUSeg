from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "outputs" / "coarse_roi_eval" / "coarse_roi_per_image_blocks.csv"
SAVE_DIR = ROOT / "outputs" / "coarse_roi_eval"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV_PATH)

plt.figure(figsize=(7, 4))
plt.hist(df["gt_roi_blocks"], bins=range(0, 58), alpha=0.6, label="GT ROI blocks")
plt.hist(df["pred_roi_blocks"], bins=range(0, 58), alpha=0.6, label="Pred ROI blocks")
plt.xlabel("Number of ROI blocks per image")
plt.ylabel("Number of images")
plt.legend()
plt.tight_layout()

save_path = SAVE_DIR / "roi_block_distribution.png"
plt.savefig(save_path, dpi=300)
print(f"saved to: {save_path}")
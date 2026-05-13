import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.coarse_roi_net import LightCoarseROINet
from data.coarse_roi_dataset import CoarseROIPixelDataset
from tools.train_coarse_roi import pixel_label_to_block_soft_targets


# ========= 路径设置 =========
IMAGE_DIR = "/root/autodl-tmp/datasets/MUSeg/test_official/Image_1024x896"
LABEL_DIR = "/root/autodl-tmp/datasets/MUSeg/test_official/Label_1024x896"
CKPT_PATH = ROOT / "checkpoints" / "coarse_roi" / "best_coarse_roi_finetune.pth"

SAVE_DIR = ROOT / "outputs" / "coarse_roi_eval"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ========= 参数设置 =========
ROI_CLASS_IDS = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
BLOCK_SIZE = 128
BATCH_SIZE = 2
PRED_THRESHOLD = 0.5


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    dataset = CoarseROIPixelDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = LightCoarseROINet(base_ch=16).to(device)
    state_dict = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    total = 0
    correct = 0
    tp = fp = fn = tn = 0
    pred_pos = 0
    gt_pos = 0

    per_image_rows = []

    for images, label_maps, stems in loader:
        images = images.to(device)
        label_maps = label_maps.to(device)

        # 与当前 train_coarse_roi.py 验证口径一致：
        # 像素级语义标签 -> 块级 soft target -> 二值参考标签
        soft_targets = pixel_label_to_block_soft_targets(
            label_maps,
            ROI_CLASS_IDS,
            block_size=BLOCK_SIZE
        )
        hard_targets = (soft_targets >= 0.5).float()

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= PRED_THRESHOLD).float()

        total += hard_targets.numel()
        correct += (preds == hard_targets).sum().item()

        tp += ((preds == 1) & (hard_targets == 1)).sum().item()
        fp += ((preds == 1) & (hard_targets == 0)).sum().item()
        fn += ((preds == 0) & (hard_targets == 1)).sum().item()
        tn += ((preds == 0) & (hard_targets == 0)).sum().item()

        pred_pos += (preds == 1).sum().item()
        gt_pos += (hard_targets == 1).sum().item()

        preds_np = preds.cpu().numpy()
        gts_np = hard_targets.cpu().numpy()

        for i, stem in enumerate(stems):
            gt_blocks = int(gts_np[i].sum())
            pred_blocks = int(preds_np[i].sum())

            per_image_rows.append([
                stem,
                gt_blocks,
                pred_blocks,
                gt_blocks / 56.0,
                pred_blocks / 56.0
            ])

    acc = correct / (total + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    gt_pos_ratio = gt_pos / (total + 1e-8)
    pred_pos_ratio = pred_pos / (total + 1e-8)

    avg_gt_blocks = gt_pos / len(dataset)
    avg_pred_blocks = pred_pos / len(dataset)

    print("\n========== Coarse ROI Test Metrics ==========")
    print(f"total blocks        = {total}")
    print(f"Accuracy            = {acc:.6f}")
    print(f"Precision           = {precision:.6f}")
    print(f"Recall              = {recall:.6f}")
    print(f"F1                  = {f1:.6f}")
    print(f"IoU                 = {iou:.6f}")
    print(f"GT ROI ratio        = {gt_pos_ratio:.6f}")
    print(f"Pred ROI ratio      = {pred_pos_ratio:.6f}")
    print(f"Avg GT ROI blocks   = {avg_gt_blocks:.4f} / 56")
    print(f"Avg Pred ROI blocks = {avg_pred_blocks:.4f} / 56")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    metrics_path = SAVE_DIR / "coarse_roi_test_metrics.csv"
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["accuracy", acc])
        writer.writerow(["precision", precision])
        writer.writerow(["recall", recall])
        writer.writerow(["f1", f1])
        writer.writerow(["iou", iou])
        writer.writerow(["gt_pos_ratio", gt_pos_ratio])
        writer.writerow(["pred_pos_ratio", pred_pos_ratio])
        writer.writerow(["avg_gt_roi_blocks", avg_gt_blocks])
        writer.writerow(["avg_pred_roi_blocks", avg_pred_blocks])
        writer.writerow(["tp", tp])
        writer.writerow(["fp", fp])
        writer.writerow(["fn", fn])
        writer.writerow(["tn", tn])

    per_image_path = SAVE_DIR / "coarse_roi_per_image_blocks.csv"
    with open(per_image_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image",
            "gt_roi_blocks",
            "pred_roi_blocks",
            "gt_roi_ratio",
            "pred_roi_ratio"
        ])
        writer.writerows(per_image_rows)

    print(f"\nmetrics saved to: {metrics_path}")
    print(f"per-image block stats saved to: {per_image_path}")


if __name__ == "__main__":
    main()
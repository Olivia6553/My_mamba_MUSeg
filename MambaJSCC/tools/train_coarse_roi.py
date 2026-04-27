import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.coarse_roi_net import LightCoarseROINet
#from data.coarse_roi_dataset import CoarseROIDataset
### 2026/4/24 wyj:改像素级训练
from data.coarse_roi_dataset import CoarseROIPixelDataset
ROI_CLASS_IDS = [1,4,5,6,7,8,9,10,11,12,13,14,15]


# def calc_pos_weight(dataset):
#     """
#     统计 ROI / BG 不平衡，给 BCEWithLogitsLoss 用
#     """
#     pos = 0.0
#     neg = 0.0
#     for _, grid, _ in dataset:
#         pos += float((grid > 0.5).sum())
#         neg += float((grid <= 0.5).sum())

#     if pos == 0:
#         return 1.0
#     return max(1.0, neg / pos)

# ### 2026/4/23 wyj:改进
# def calc_pos_weight(dataset, name="dataset"):
#     pos = 0.0
#     neg = 0.0
#     for _, grid, _ in dataset:
#         pos += float((grid > 0.5).sum())
#         neg += float((grid <= 0.5).sum())

#     total = pos + neg
#     pos_ratio = pos / (total + 1e-8)
#     neg_ratio = neg / (total + 1e-8)
#     ratio = neg / (pos + 1e-8)

#     print(f"[{name}] pos={pos:.0f}, neg={neg:.0f}, pos_ratio={pos_ratio:.4f}, neg_ratio={neg_ratio:.4f}, neg/pos={ratio:.4f}")

#     return max(1.0, ratio)

### 2026/4/24 wyj:改像素级训练
def calc_pos_weight(dataset, roi_class_ids, block_size=128, name="dataset"):
    pos = 0.0
    neg = 0.0

    for _, label_map, _ in dataset:
        label_map = label_map.unsqueeze(0)  # [1,H,W]
        soft_targets = pixel_label_to_block_soft_targets(label_map, roi_class_ids, block_size=block_size)
        hard_targets = (soft_targets >= 0.5).float()

        pos += float((hard_targets > 0.5).sum())
        neg += float((hard_targets <= 0.5).sum())

    total = pos + neg
    pos_ratio = pos / (total + 1e-8)
    neg_ratio = neg / (total + 1e-8)
    ratio = neg / (pos + 1e-8)

    print(f"[{name}] pos={pos:.0f}, neg={neg:.0f}, pos_ratio={pos_ratio:.4f}, neg_ratio={neg_ratio:.4f}, neg/pos={ratio:.4f}")

    return max(1.0, ratio)


### 2026/4/24 wyj:改像素级训练
def pixel_label_to_block_soft_targets(label_map, roi_class_ids, block_size=128):
    """
    输入:
        label_map: [B, H, W]，像素级语义标签图
    输出:
        soft_targets: [B, 1, 7, 8]，每个块的 ROI 占比，范围 [0,1]
    """
    device = label_map.device
    roi_ids = torch.tensor(roi_class_ids, device=device, dtype=label_map.dtype)

    # 像素级 ROI mask: [B,H,W]
    roi_mask = (label_map.unsqueeze(-1) == roi_ids).any(dim=-1).float()

    B, H, W = roi_mask.shape
    gh = H // block_size
    gw = W // block_size

    # 按块展开并求每块 ROI 占比
    blocks = roi_mask.unfold(1, block_size, block_size).unfold(2, block_size, block_size)
    # [B, gh, gw, block, block]
    soft_targets = blocks.contiguous().mean(dim=(-1, -2))   # [B, gh, gw]

    return soft_targets.unsqueeze(1)  # [B,1,7,8]




@torch.no_grad()
# def evaluate(model, loader, device, threshold=0.35):
#     """
#     保守型二分类:
#     推理阈值设置低一点，更容易判成 ROI
#     """
#     model.eval()

#     total = 0
#     correct = 0
#     tp = 0
#     fn = 0

#     for images, targets, _ in loader:
#         images = images.to(device)
#         targets = targets.to(device)

#         logits = model(images)
#         probs = torch.sigmoid(logits)
#         preds = (probs >= threshold).float()

#         total += targets.numel()
#         correct += (preds == targets).sum().item()
#         tp += ((preds == 1) & (targets == 1)).sum().item()
#         fn += ((preds == 0) & (targets == 1)).sum().item()

#     acc = correct / total
#     roi_recall = tp / (tp + fn + 1e-8)
#     return acc, roi_recall

# ### 2026/4/23 wyj:改进
# @torch.no_grad()
# def evaluate(model, loader, device, threshold=0.5):
#     model.eval()

#     total = 0
#     correct = 0

#     tp = 0
#     fp = 0
#     fn = 0
#     tn = 0

#     pred_pos = 0
#     gt_pos = 0

#     for images, targets, _ in loader:
#         images = images.to(device)
#         targets = targets.to(device)

#         logits = model(images)
#         probs = torch.sigmoid(logits)
#         preds = (probs >= threshold).float()

#         total += targets.numel()
#         correct += (preds == targets).sum().item()

#         tp += ((preds == 1) & (targets == 1)).sum().item()
#         fp += ((preds == 1) & (targets == 0)).sum().item()
#         fn += ((preds == 0) & (targets == 1)).sum().item()
#         tn += ((preds == 0) & (targets == 0)).sum().item()

#         pred_pos += (preds == 1).sum().item()
#         gt_pos += (targets == 1).sum().item()

#     acc = correct / (total + 1e-8)
#     recall = tp / (tp + fn + 1e-8)
#     precision = tp / (tp + fp + 1e-8)
#     f1 = 2 * precision * recall / (precision + recall + 1e-8)
#     iou = tp / (tp + fp + fn + 1e-8)

#     pred_pos_ratio = pred_pos / (total + 1e-8)
#     gt_pos_ratio = gt_pos / (total + 1e-8)

#     return {
#         "acc": acc,
#         "recall": recall,
#         "precision": precision,
#         "f1": f1,
#         "iou": iou,
#         "pred_pos_ratio": pred_pos_ratio,
#         "gt_pos_ratio": gt_pos_ratio,
#     }

### 2026/4/24 wyj:改像素级训练
@torch.no_grad()
def evaluate(model, loader, device, roi_class_ids, block_size=128, threshold=0.5):
    model.eval()

    total = 0
    correct = 0

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    pred_pos = 0
    gt_pos = 0

    for images, label_maps, _ in loader:
        images = images.to(device)
        label_maps = label_maps.to(device)

        soft_targets = pixel_label_to_block_soft_targets(label_maps, roi_class_ids, block_size=block_size)
        hard_targets = (soft_targets >= 0.5).float()

        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()

        total += hard_targets.numel()
        correct += (preds == hard_targets).sum().item()

        tp += ((preds == 1) & (hard_targets == 1)).sum().item()
        fp += ((preds == 1) & (hard_targets == 0)).sum().item()
        fn += ((preds == 0) & (hard_targets == 1)).sum().item()
        tn += ((preds == 0) & (hard_targets == 0)).sum().item()

        pred_pos += (preds == 1).sum().item()
        gt_pos += (hard_targets == 1).sum().item()

    acc = correct / (total + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    pred_pos_ratio = pred_pos / (total + 1e-8)
    gt_pos_ratio = gt_pos / (total + 1e-8)

    return {
        "acc": acc,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "iou": iou,
        "pred_pos_ratio": pred_pos_ratio,
        "gt_pos_ratio": gt_pos_ratio,
    }


# def train_one_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0

#     # for images, targets, _ in loader:
#     for images, label_maps, _ in loader:
#         images = images.to(device)
#         targets = targets.to(device)

#         logits = model(images)
#         loss = criterion(logits, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / max(1, len(loader))

### 2026/4/24 wyj:改像素级训练
def train_one_epoch(model, loader, optimizer, criterion, device, roi_class_ids, block_size=128):
    model.train()
    total_loss = 0.0

    for images, label_maps, _ in loader:
        images = images.to(device)
        label_maps = label_maps.to(device)

        soft_targets = pixel_label_to_block_soft_targets(label_maps, roi_class_ids, block_size=block_size)

        logits = model(images)
        loss = criterion(logits, soft_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))


def main():
    # ===== 数据路径 =====
    train_image_dir = "/home/wengyijia/datasets/MUSeg/train_official/Image_1024x896"
    #train_grid_dir  = "/home/wengyijia/datasets/MUSeg/train/BlockROI_binary_grid_7x8"

    val_image_dir = "/home/wengyijia/datasets/MUSeg/test_official/Image_1024x896"
    #val_grid_dir  = "/home/wengyijia/datasets/MUSeg/val/BlockROI_binary_grid_7x8"

### 2026/4/24：wyj 像素级分割数据
    train_label_dir = "/home/wengyijia/datasets/MUSeg/train_official/Label_1024x896"
    val_label_dir   = "/home/wengyijia/datasets/MUSeg/test_official/Label_1024x896"

    # ===== 保存路径 =====
    save_dir = ROOT / "checkpoints" / "coarse_roi"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ===== 数据集 =====
    # train_set = CoarseROIDataset(train_image_dir, train_grid_dir)
    # val_set = CoarseROIDataset(val_image_dir, val_grid_dir)
    train_set = CoarseROIPixelDataset(train_image_dir, train_label_dir)
    val_set = CoarseROIPixelDataset(val_image_dir, val_label_dir)

    train_loader = DataLoader(train_set, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=2, shuffle=False, num_workers=0)

    # ===== 设备 =====
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    # ===== 模型 =====
    model = LightCoarseROINet(base_ch=16).to(device)

    # ===== 类别不平衡处理 =====
    # pos_weight_value = calc_pos_weight(train_set)
    # print("pos_weight =", pos_weight_value)

    # criterion = nn.BCEWithLogitsLoss(
    #     pos_weight=torch.tensor([pos_weight_value], dtype=torch.float32, device=device)
    # )

### 2026/4/23 wyj:改进
    # train_pos_weight = calc_pos_weight(train_set, name="train")
    # _ = calc_pos_weight(val_set, name="val")
    train_pos_weight = calc_pos_weight(train_set, ROI_CLASS_IDS, block_size=128, name="train")
    _ = calc_pos_weight(val_set, ROI_CLASS_IDS, block_size=128, name="val")
    print("train pos_weight =", train_pos_weight)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([train_pos_weight], dtype=torch.float32, device=device)
    )



    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)

    # ===== 训练 =====
    best_f1 = -1.0
    epochs = 20

    for epoch in range(1, epochs + 1):
        # train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # val_acc, val_roi_recall = evaluate(model, val_loader, device, threshold=0.35)

        # print(
        #     f"Epoch [{epoch:02d}/{epochs}] "
        #     f"train_loss={train_loss:.4f} "
        #     f"val_acc={val_acc:.4f} "
        #     f"val_roi_recall={val_roi_recall:.4f}"
        # )

        # # 你更关心 ROI 别漏掉，所以按 ROI recall 保存
        # if val_roi_recall > best_recall:
        #     best_recall = val_roi_recall
        #     save_path = save_dir / "best_coarse_roi.pth"
        #     torch.save(model.state_dict(), save_path)
        #     print(f"[保存最优模型] {save_path}")

### 2026/4/23 wyj:改进
        # train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        # metrics = evaluate(model, val_loader, device, threshold=0.5)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
            roi_class_ids=ROI_CLASS_IDS, block_size=128
        )

        metrics = evaluate(
            model, val_loader, device,
            roi_class_ids=ROI_CLASS_IDS, block_size=128, threshold=0.5
        )

        print(
            f"Epoch [{epoch:02d}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_acc={metrics['acc']:.4f} "
            f"val_precision={metrics['precision']:.4f} "
            f"val_recall={metrics['recall']:.4f} "
            f"val_f1={metrics['f1']:.4f} "
            f"val_iou={metrics['iou']:.4f} "
            f"pred_pos_ratio={metrics['pred_pos_ratio']:.4f} "
            f"gt_pos_ratio={metrics['gt_pos_ratio']:.4f}"
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            save_path = save_dir / "best_coarse_roi.pth"
            torch.save(model.state_dict(), save_path)
            print(f"[保存最优模型] {save_path}")


if __name__ == "__main__":
    main()
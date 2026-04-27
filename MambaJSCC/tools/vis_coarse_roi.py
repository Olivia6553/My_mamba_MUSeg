import os
import sys
import random
import inspect
import importlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

# =========================================================
# 让脚本能找到项目根目录下的 data / models
# =========================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.coarse_roi_dataset import CoarseROIPixelDataset

# =========================================================
# 基本配置
# =========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_DIR = "/home/wengyijia/datasets/MUSeg/test_official/Image_1024x896"
LABEL_DIR = "/home/wengyijia/datasets/MUSeg/test_official/Label_1024x896"

#CKPT_PATH = "/home/wengyijia/mambajscc/MambaJSCC/checkpoints/coarse_roi/best_coarse_roi.pth"
CKPT_PATH = "/home/wengyijia/mambajscc/MambaJSCC/checkpoints/coarse_roi/best_coarse_roi_finetune.pth"

SAVE_DIR = "/home/wengyijia/mambajscc/MambaJSCC/vis_results/2_finetune_coarse_roi_samples"
os.makedirs(SAVE_DIR, exist_ok=True)

THRESHOLD = 0.5
SEED = 2048

# 分层抽样数量
NUM_LOW = 5
NUM_MID = 5
NUM_HIGH = 5

# 和训练时保持一致
ROI_CLASS_IDS =[1,4,5,6,7,8,9,10,11,12,13,14,15]

IMG_H = 896
IMG_W = 1024
BLOCK_H = 128
BLOCK_W = 128
GRID_H = IMG_H // BLOCK_H   # 7
GRID_W = IMG_W // BLOCK_W   # 8


# =========================================================
# 工具函数
# =========================================================
def tensor_img_to_uint8(img_tensor):
    """
    img_tensor: [3, H, W], torch.Tensor, 值域[0,1]
    返回: uint8 RGB ndarray, [H,W,3]
    """
    img = img_tensor.detach().cpu().float().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)
    return img


def pixel_label_to_block_gt(label_map):
    """
    label_map: [H, W], 像素级语义标签图
    返回:
        gt_block: [7, 8], 0/1
    """
    roi_mask = np.isin(label_map, ROI_CLASS_IDS).astype(np.float32)

    blocks = roi_mask.reshape(
        GRID_H, BLOCK_H,
        GRID_W, BLOCK_W
    ).transpose(0, 2, 1, 3)  # [7,8,128,128]

    block_ratio = blocks.mean(axis=(2, 3))   # [7,8]
    gt_block = (block_ratio >= 0.5).astype(np.float32)
    return gt_block


def block_to_full(block_map):
    """
    block_map: [7,8]
    返回: [896,1024]
    """
    return np.repeat(np.repeat(block_map, BLOCK_H, axis=0), BLOCK_W, axis=1)


def overlay_mask_on_image(img_uint8, mask01, alpha=0.35):
    """
    img_uint8: [H,W,3], uint8
    mask01:    [H,W], 0/1
    ROI 区域叠加红色
    """
    img = img_uint8.astype(np.float32)
    red = np.zeros_like(img, dtype=np.float32)
    red[..., 0] = 255.0

    mask3 = np.stack([mask01, mask01, mask01], axis=-1).astype(np.float32)
    out = np.where(mask3 > 0, (1 - alpha) * img + alpha * red, img)
    return np.clip(out, 0, 255).astype(np.uint8)


def prob_to_heatmap(prob_big):
    """
    prob_big: [H,W], 0~1
    简单手写一个蓝->青->黄->红的热力图，不依赖 matplotlib
    返回: [H,W,3], uint8
    """
    p = np.clip(prob_big, 0.0, 1.0)

    r = np.zeros_like(p)
    g = np.zeros_like(p)
    b = np.zeros_like(p)

    # 0~0.33: 蓝 -> 青
    m1 = (p < 0.33)
    t1 = p[m1] / 0.33
    r[m1] = 0
    g[m1] = 255 * t1
    b[m1] = 255

    # 0.33~0.66: 青 -> 黄
    m2 = (p >= 0.33) & (p < 0.66)
    t2 = (p[m2] - 0.33) / 0.33
    r[m2] = 255 * t2
    g[m2] = 255
    b[m2] = 255 * (1 - t2)

    # 0.66~1.0: 黄 -> 红
    m3 = (p >= 0.66)
    t3 = (p[m3] - 0.66) / 0.34
    r[m3] = 255
    g[m3] = 255 * (1 - t3)
    b[m3] = 0

    heat = np.stack([r, g, b], axis=-1)
    return np.clip(heat, 0, 255).astype(np.uint8)


def draw_grid_lines(img_uint8, color=(255, 255, 255), width=2):
    """
    在图上画 7x8 网格线
    """
    pil = Image.fromarray(img_uint8)
    draw = ImageDraw.Draw(pil)

    # 横线
    for i in range(1, GRID_H):
        y = i * BLOCK_H
        draw.line([(0, y), (IMG_W, y)], fill=color, width=width)

    # 竖线
    for j in range(1, GRID_W):
        x = j * BLOCK_W
        draw.line([(x, 0), (x, IMG_H)], fill=color, width=width)

    return np.array(pil)


def add_title(img_uint8, title):
    """
    在图上方加一个简单标题栏
    """
    title_h = 40
    canvas = Image.new("RGB", (img_uint8.shape[1], img_uint8.shape[0] + title_h), (20, 20, 20))
    img = Image.fromarray(img_uint8)
    canvas.paste(img, (0, title_h))

    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 255, 255))

    return np.array(canvas)


def concat_h(imgs):
    """
    横向拼接，要求高度相同
    """
    return np.concatenate(imgs, axis=1)


def concat_v(imgs):
    """
    纵向拼接，要求宽度相同
    """
    return np.concatenate(imgs, axis=0)


def save_rgb(img_uint8, save_path):
    Image.fromarray(img_uint8).save(save_path)


def get_sample_name(stem_or_idx):
    return str(stem_or_idx)


# =========================================================
# 模型加载：尽量自动适配你的 coarse_roi_net.py
# =========================================================
def build_model():
    module = importlib.import_module("models.coarse_roi_net")

    candidate_names = [
        "CoarseROINet",
        "LightCoarseROINet",
    ]

    model_cls = None
    for name in candidate_names:
        if hasattr(module, name):
            model_cls = getattr(module, name)
            print(f"[INFO] 使用模型类: {name}")
            break

    if model_cls is None:
        avail = [x for x in dir(module) if x.endswith("Net") or x.endswith("Model")]
        raise RuntimeError(
            f"在 models/coarse_roi_net.py 里没找到常见模型类名。\n"
            f"尝试过: {candidate_names}\n"
            f"当前可见候选: {avail}"
        )

    # 尝试几种构造方式
    init_trials = [
        {},
        {"base_ch": 16},
        {"base_ch": 16, "grid_h": 7, "grid_w": 8},
    ]

    model = None
    last_err = None
    for kwargs in init_trials:
        try:
            model = model_cls(**kwargs)
            print(f"[INFO] 模型初始化成功，kwargs={kwargs}")
            break
        except Exception as e:
            last_err = e

    if model is None:
        raise RuntimeError(f"模型初始化失败，最后一次错误: {last_err}")

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


# =========================================================
# 抽样
# =========================================================
def choose_stratified_indices(dataset, num_low=3, num_mid=3, num_high=3, seed=1024):
    """
    按 GT 块级 ROI 占比分层抽样
    low:  < 0.10
    mid:  [0.10, 0.30)
    high: >= 0.30
    """
    random.seed(seed)

    low_pool = []
    mid_pool = []
    high_pool = []

    for i in range(len(dataset)):
        _, label_map, _ = dataset[i]
        label_map = label_map.numpy()              # [896,1024]
        gt_block = pixel_label_to_block_gt(label_map)
        ratio = gt_block.mean()

        if ratio < 0.10:
            low_pool.append(i)
        elif ratio < 0.30:
            mid_pool.append(i)
        else:
            high_pool.append(i)

    print(f"low_pool  (<0.10)      = {len(low_pool)}")
    print(f"mid_pool  (0.10~0.30)  = {len(mid_pool)}")
    print(f"high_pool (>=0.30)     = {len(high_pool)}")

    chosen = []
    if len(low_pool) > 0:
        chosen += random.sample(low_pool, min(num_low, len(low_pool)))
    if len(mid_pool) > 0:
        chosen += random.sample(mid_pool, min(num_mid, len(mid_pool)))
    if len(high_pool) > 0:
        chosen += random.sample(high_pool, min(num_high, len(high_pool)))

    return chosen


# =========================================================
# 可视化
# =========================================================
def visualize_one_sample(model, dataset, idx, threshold=0.5):
    x, label_map, stem = dataset[idx]

    label_map = label_map.numpy()                  # [896,1024]
    gt_block = pixel_label_to_block_gt(label_map)  # [7,8]

    x_in = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x_in)                       # [1,1,7,8]
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    pred_block = (probs >= threshold).astype(np.float32)

    img_uint8 = tensor_img_to_uint8(x)

    gt_big = block_to_full(gt_block)
    pred_big = block_to_full(pred_block)
    prob_big = block_to_full(probs)

    gt_overlay = overlay_mask_on_image(img_uint8, gt_big, alpha=0.35)
    pred_overlay = overlay_mask_on_image(img_uint8, pred_big, alpha=0.35)
    heatmap = prob_to_heatmap(prob_big)

    # 加网格线
    img_show = draw_grid_lines(img_uint8)
    gt_show = draw_grid_lines(gt_overlay)
    pred_show = draw_grid_lines(pred_overlay)
    heat_show = draw_grid_lines(heatmap)

    gt_ratio = gt_block.mean()
    pred_ratio = pred_block.mean()

    img_show = add_title(img_show, f"Original | {stem}")
    gt_show = add_title(gt_show, f"GT Block ROI | ratio={gt_ratio:.3f}")
    pred_show = add_title(pred_show, f"Pred Block ROI | ratio={pred_ratio:.3f} | thr={threshold}")
    heat_show = add_title(heat_show, "Pred Probability Heatmap")

    top = concat_h([img_show, gt_show])
    bottom = concat_h([pred_show, heat_show])
    canvas = concat_v([top, bottom])

    save_path = os.path.join(SAVE_DIR, f"{stem}.png")
    save_rgb(canvas, save_path)
    print(f"[保存] {save_path}")


def main():
    print("device =", DEVICE)

    dataset = CoarseROIPixelDataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
    )

    model = build_model()

    chosen_indices = choose_stratified_indices(
        dataset,
        num_low=NUM_LOW,
        num_mid=NUM_MID,
        num_high=NUM_HIGH,
        seed=SEED
    )

    print("chosen_indices =", chosen_indices)

    for idx in chosen_indices:
        visualize_one_sample(model, dataset, idx, threshold=THRESHOLD)

    print("\n全部可视化完成。")
    print(f"输出目录: {SAVE_DIR}")


if __name__ == "__main__":
    main()
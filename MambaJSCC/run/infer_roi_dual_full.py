from utils.complexity import (
    count_params,
    to_million,
    to_giga,
    profile_roi_head_macs,
    profile_branch_macs_per_block,
    timer_start,
    timer_end_ms,
)

import os
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from models.coarse_roi_net import LightCoarseROINet
from models.channel import Channel
from utils.distortion import *
from utils.utils import *


def load_image_as_tensor(img_path):
    """
    return:
        img_tensor: [1,3,896,1024], float32, range [0,1]
        img_pil
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_np = np.asarray(img_pil, dtype=np.float32) / 255.0  # [H,W,3]
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor, img_pil


def save_tensor_image(tensor, save_path):
    """
    tensor: [1,3,H,W] or [3,H,W]
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = tensor.detach().cpu().clamp(0, 1)

    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    vutils.save_image(tensor, str(save_path))


def calc_psnr(x, y):
    """
    x,y: [1,3,H,W], range [0,1]
    """
    mse = torch.mean((x - y) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10(1.0 / mse)


def calc_region_psnr(x, y, mask_2d):
    """
    x,y: [1,3,H,W]
    mask_2d: [H,W], 0/1
    """
    mask = torch.from_numpy(mask_2d).float().to(x.device)  # [H,W]
    mask = mask.unsqueeze(0).unsqueeze(0)                  # [1,1,H,W]

    if mask.sum().item() < 1:
        return None

    mse = (((x - y) ** 2) * mask).sum() / (mask.sum() * x.shape[1] + 1e-8)
    mse = mse.item()

    if mse <= 1e-12:
        return 99.0

    return 10.0 * np.log10(1.0 / mse)


def grid_to_full_mask(grid, block_h=128, block_w=128):
    """
    grid: [7,8], 0/1
    return: [896,1024], 0/1
    """
    full = np.repeat(np.repeat(grid.astype(np.uint8), block_h, axis=0), block_w, axis=1)
    return full


def save_roi_grid_big(grid, save_path, block_h=128, block_w=128):
    """
    保存 1024×896 的块状 ROI 图
    ROI=255, BG=0
    """
    full = grid_to_full_mask(grid, block_h, block_w) * 255
    img = Image.fromarray(full.astype(np.uint8), mode="L")
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(save_path)


def save_overlay_roi(img_pil, grid, save_path, alpha=0.35):
    """
    在原图上叠加 ROI 块，ROI 区域显示红色半透明
    """
    img = np.asarray(img_pil.convert("RGB"), dtype=np.float32)
    mask = grid_to_full_mask(grid)  # [H,W]

    overlay = img.copy()
    red = np.zeros_like(img)
    red[..., 0] = 255

    overlay[mask == 1] = (1 - alpha) * img[mask == 1] + alpha * red[mask == 1]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(save_path)


def split_into_blocks(img_tensor, block_h=128, block_w=128):
    """
    img_tensor: [1,3,896,1024]
    return:
        blocks: list of [3,128,128]
        positions: list of (r,c)
    """
    _, _, H, W = img_tensor.shape

    assert H % block_h == 0 and W % block_w == 0, f"image size {H}x{W} cannot divide by {block_h}x{block_w}"

    gh = H // block_h
    gw = W // block_w

    blocks = []
    positions = []

    for r in range(gh):
        for c in range(gw):
            y0 = r * block_h
            y1 = (r + 1) * block_h
            x0 = c * block_w
            x1 = (c + 1) * block_w

            block = img_tensor[0, :, y0:y1, x0:x1]  # [3,128,128]
            blocks.append(block)
            positions.append((r, c))

    return blocks, positions


def merge_blocks(block_dict, H=896, W=1024, block_h=128, block_w=128, device="cuda"):
    """
    block_dict:
        key: (r,c)
        value: [3,128,128]
    return:
        recon_full: [1,3,H,W]
    """
    recon = torch.zeros(1, 3, H, W, device=device)

    for (r, c), block in block_dict.items():
        y0 = r * block_h
        y1 = (r + 1) * block_h
        x0 = c * block_w
        x1 = (c + 1) * block_w

        recon[0, :, y0:y1, x0:x1] = block

    return recon


@torch.no_grad()
def run_branch_codec(blocks_tensor, encoder, decoder, channel, config, SNR):
    """
    blocks_tensor: [N,3,128,128]
    return:
        recon_blocks: [N,3,128,128]
        feature_numel: encoder 输出元素数，用于统计 CBR
    """
    if blocks_tensor is None or blocks_tensor.shape[0] == 0:
        return None, 0

    feature = encoder(blocks_tensor, SNR)
    feature_numel = feature.numel()

    received, pwr, h = channel.forward(feature, SNR)

    if config.CHANNEL.TYPE == "rayleigh":
        sigma_square = 1.0 / (10 ** (SNR / 10))
        received = torch.conj(h) * received / (torch.abs(h) ** 2 + sigma_square)
    elif config.CHANNEL.TYPE == "awgn":
        pass
    else:
        raise ValueError(f"Unknown channel type: {config.CHANNEL.TYPE}")

    received = torch.cat((torch.real(received), torch.imag(received)), dim=2) * torch.sqrt(pwr)
    recon_blocks = decoder(received, SNR)

    return recon_blocks.clamp(0, 1), feature_numel


def build_branch_ckpt_name(config, branch_type):
    encoder_name = (
        config.TRAIN.ENCODER_PATH
        + f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}"
        + f"_depth{config.MODEL.VSSM.DEPTHS}"
        + f"_embed{config.MODEL.VSSM.EMBED_DIM}"
        + f"_rsl{config.DATA.IMG_SIZE}.pt"
    )

    decoder_name = (
        config.TRAIN.DECODER_PATH
        + f"{branch_type}_OUTCHANS{config.MODEL.VSSM.OUT_CHANS}"
        + f"_depth{config.MODEL.VSSM.DEPTHS}"
        + f"_embed{config.MODEL.VSSM.EMBED_DIM}"
        + f"_rsl{config.DATA.IMG_SIZE}.pt"
    )

    return encoder_name, decoder_name


@torch.no_grad()
def infer_roi_dual_full(
    roi_config,
    bg_config,
    image_dir="/root/autodl-tmp/datasets/MUSeg/test_official/Image_1024x896",
    coarse_roi_ckpt="/root/autodl-tmp/mambajscc/MambaJSCC/checkpoints/coarse_roi/best_coarse_roi_finetune.pth",
    output_dir="/root/autodl-tmp/mambajscc/MambaJSCC/outputs/roi_dual_full",
    threshold=0.5,
    image_start=0,
    max_images=20,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visual_dir = output_dir / "visual"
    visual_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metrics.csv"

    # ===== 1. 加载粗 ROI 模块 =====
    roi_head = LightCoarseROINet(base_ch=16).to(device)

    coarse_roi_ckpt = Path(coarse_roi_ckpt)
    if not coarse_roi_ckpt.exists():
        fallback = Path("/root/autodl-tmp/mambajscc/MambaJSCC/checkpoints/coarse_roi/best_coarse_roi.pth")
        if fallback.exists():
            coarse_roi_ckpt = fallback
        else:
            raise FileNotFoundError(f"找不到 ROI 头 checkpoint: {coarse_roi_ckpt}")

    roi_head.load_state_dict(torch.load(coarse_roi_ckpt, map_location=device))
    roi_head.eval()

    print(f"[Load] coarse ROI ckpt = {coarse_roi_ckpt}")

    # ===== 2. 加载 ROI 分支 =====
    roi_encoder_name, roi_decoder_name = build_branch_ckpt_name(roi_config, "roi")
    print(f"[Load] ROI encoder = {roi_encoder_name}")
    print(f"[Load] ROI decoder = {roi_decoder_name}")

    roi_encoder = torch.load(roi_encoder_name, map_location="cpu").to(device)
    roi_decoder = torch.load(roi_decoder_name, map_location="cpu").to(device)
    roi_encoder.eval()
    roi_decoder.eval()
    roi_channel = Channel(roi_config)

    # ===== 3. 加载 BG 分支 =====
    bg_encoder_name, bg_decoder_name = build_branch_ckpt_name(bg_config, "bg")
    print(f"[Load] BG encoder = {bg_encoder_name}")
    print(f"[Load] BG decoder = {bg_decoder_name}")

    bg_encoder = torch.load(bg_encoder_name, map_location="cpu").to(device)
    bg_decoder = torch.load(bg_decoder_name, map_location="cpu").to(device)
    bg_encoder.eval()
    bg_decoder.eval()
    bg_channel = Channel(bg_config)


    ####

    # ===== 复杂度静态统计：Params + MACs =====
    roi_head_params = count_params(roi_head)
    roi_branch_params = count_params(roi_encoder) + count_params(roi_decoder)
    bg_branch_params = count_params(bg_encoder) + count_params(bg_decoder)

    total_stored_params = roi_head_params + roi_branch_params + bg_branch_params

    SNR_roi_profile = roi_config.CHANNEL.SNR[0]
    SNR_bg_profile = bg_config.CHANNEL.SNR[0]

    print("\n========== Static Complexity ==========")
    print(f"ROI head params      = {to_million(roi_head_params):.4f} M")
    print(f"ROI branch params    = {to_million(roi_branch_params):.4f} M")
    print(f"BG branch params     = {to_million(bg_branch_params):.4f} M")
    print(f"Total stored params  = {to_million(total_stored_params):.4f} M")

    roi_head_macs = profile_roi_head_macs(
        roi_head=roi_head,
        device=device,
        h=896,
        w=1024,
    )

    roi_branch_macs_dict = profile_branch_macs_per_block(
        encoder=roi_encoder,
        decoder=roi_decoder,
        channel=roi_channel,
        config=roi_config,
        device=device,
        snr=SNR_roi_profile,
        block_h=128,
        block_w=128,
    )

    bg_branch_macs_dict = profile_branch_macs_per_block(
        encoder=bg_encoder,
        decoder=bg_decoder,
        channel=bg_channel,
        config=bg_config,
        device=device,
        snr=SNR_bg_profile,
        block_h=128,
        block_w=128,
    )

    print(f"ROI head MACs        = {to_giga(roi_head_macs):.4f} G")
    print(f"ROI branch MACs/block= {to_giga(roi_branch_macs_dict['branch_macs']):.4f} G")
    print(f"BG branch MACs/block = {to_giga(bg_branch_macs_dict['branch_macs']):.4f} G")


    # image_paths = sorted([
    #     p for p in image_dir.iterdir()
    #     if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    # ])

    # if max_images is not None:
    #     image_paths = image_paths[:max_images]

    image_paths = sorted([
        p for p in image_dir.iterdir()
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    # 从第 image_start 张图片开始取
    if image_start > 0:
        image_paths = image_paths[image_start:]

    # 再取 max_images 张
    if max_images is not None:
        image_paths = image_paths[:max_images]

    print(f"[Infer] image_dir = {image_dir}")
    print(f"[Infer] image_start = {image_start}")
    print(f"[Infer] num_images = {len(image_paths)}")
    print(f"[Infer] output_dir = {output_dir}")
    print(f"[Infer] threshold = {threshold}")

    rows = []

    for img_path in tqdm(image_paths):
        img_tensor_cpu, img_pil = load_image_as_tensor(img_path)
        img_tensor = img_tensor_cpu.to(device)  # [1,3,896,1024]

        total_timer = timer_start()

        _, _, H, W = img_tensor.shape
        assert H == 896 and W == 1024, f"当前脚本假设输入是 896x1024，但得到 {H}x{W}"

        # ===== 4. ROI 头预测 7×8 mask =====
        # logits = roi_head(img_tensor)
        # probs = torch.sigmoid(logits)[0, 0]  # [7,8]
        # grid = (probs >= threshold).float().cpu().numpy().astype(np.uint8)  # [7,8]
        
        # ===== ROI head 推理计时 =====
        t0 = timer_start()
        logits = roi_head(img_tensor)
        roi_head_time_ms = timer_end_ms(t0)

        probs = torch.sigmoid(logits)[0, 0]
        grid = (probs >= threshold).float().cpu().numpy().astype(np.uint8)


        # ===== 5. 切 56 个块并按 grid 路由 =====
        # blocks, positions = split_into_blocks(img_tensor)

        # roi_blocks = []
        # roi_positions = []
        # bg_blocks = []
        # bg_positions = []

        # ===== 切块 + 路由计时 =====
        t0 = timer_start()

        blocks, positions = split_into_blocks(img_tensor)

        roi_blocks = []
        roi_positions = []
        bg_blocks = []
        bg_positions = []

        for block, (r, c) in zip(blocks, positions):
            if grid[r, c] == 1:
                roi_blocks.append(block)
                roi_positions.append((r, c))
            else:
                bg_blocks.append(block)
                bg_positions.append((r, c))

        roi_blocks_tensor = torch.stack(roi_blocks, dim=0).to(device) if len(roi_blocks) > 0 else None
        bg_blocks_tensor = torch.stack(bg_blocks, dim=0).to(device) if len(bg_blocks) > 0 else None

        routing_time_ms = timer_end_ms(t0)



        for block, (r, c) in zip(blocks, positions):
            if grid[r, c] == 1:
                roi_blocks.append(block)
                roi_positions.append((r, c))
            else:
                bg_blocks.append(block)
                bg_positions.append((r, c))

        roi_blocks_tensor = torch.stack(roi_blocks, dim=0).to(device) if len(roi_blocks) > 0 else None
        bg_blocks_tensor = torch.stack(bg_blocks, dim=0).to(device) if len(bg_blocks) > 0 else None

        SNR_roi = roi_config.CHANNEL.SNR[0]
        SNR_bg = bg_config.CHANNEL.SNR[0]

        recon_dict = {}
        total_feature_numel = 0

        # ===== 6. ROI 块走 ROI 分支 =====
        # roi_feature_numel = 0
        # if roi_blocks_tensor is not None:
        #     roi_recon_blocks, roi_feature_numel = run_branch_codec(
        #         roi_blocks_tensor,
        #         roi_encoder,
        #         roi_decoder,
        #         roi_channel,
        #         roi_config,
        #         SNR_roi,
        #     )
        #     total_feature_numel += roi_feature_numel

        #     for i, pos in enumerate(roi_positions):
        #         recon_dict[pos] = roi_recon_blocks[i]


        roi_feature_numel = 0
        roi_branch_time_ms = 0.0

        if roi_blocks_tensor is not None:
            t0 = timer_start()

            roi_recon_blocks, roi_feature_numel = run_branch_codec(
                roi_blocks_tensor,
                roi_encoder,
                roi_decoder,
                roi_channel,
                roi_config,
                SNR_roi,
            )

            roi_branch_time_ms = timer_end_ms(t0)

            total_feature_numel += roi_feature_numel

            for i, pos in enumerate(roi_positions):
                recon_dict[pos] = roi_recon_blocks[i]


        # ===== 7. BG 块走 BG 分支 =====
        # bg_feature_numel = 0
        # if bg_blocks_tensor is not None:
        #     bg_recon_blocks, bg_feature_numel = run_branch_codec(
        #         bg_blocks_tensor,
        #         bg_encoder,
        #         bg_decoder,
        #         bg_channel,
        #         bg_config,
        #         SNR_bg,
        #     )
        #     total_feature_numel += bg_feature_numel

        #     for i, pos in enumerate(bg_positions):
        #         recon_dict[pos] = bg_recon_blocks[i]

        bg_feature_numel = 0
        bg_branch_time_ms = 0.0

        if bg_blocks_tensor is not None:
            t0 = timer_start()

            bg_recon_blocks, bg_feature_numel = run_branch_codec(
                bg_blocks_tensor,
                bg_encoder,
                bg_decoder,
                bg_channel,
                bg_config,
                SNR_bg,
            )

            bg_branch_time_ms = timer_end_ms(t0)

            total_feature_numel += bg_feature_numel

            for i, pos in enumerate(bg_positions):
                recon_dict[pos] = bg_recon_blocks[i]



        # ===== 8. 拼回整图 =====
        # recon_full = merge_blocks(recon_dict, H=H, W=W, device=device)
            # ===== 拼接计时 =====
        t0 = timer_start()
        recon_full = merge_blocks(recon_dict, H=H, W=W, device=device)
        stitching_time_ms = timer_end_ms(t0)

        total_infer_time_ms = timer_end_ms(total_timer)


        # ===== 9. 指标计算 =====
        full_psnr = calc_psnr(img_tensor, recon_full)

        full_mask_roi = grid_to_full_mask(grid)          # [896,1024], ROI=1
        full_mask_bg = 1 - full_mask_roi                 # [896,1024], BG=1

        roi_psnr = calc_region_psnr(img_tensor, recon_full, full_mask_roi)
        bg_psnr = calc_region_psnr(img_tensor, recon_full, full_mask_bg)

        # 这里沿用你项目里的 CBR 定义:
        # CBR = feature.numel() / input_image.numel() / 2
        total_cbr = total_feature_numel / img_tensor.numel() / 2
        compression_ratio = 1.0 / (total_cbr + 1e-12)

        roi_count = int(grid.sum())
        bg_count = int(56 - roi_count)

        # ===== 整图有效 MACs =====
        effective_macs = (
            roi_head_macs
            + roi_count * roi_branch_macs_dict["branch_macs"]
            + bg_count * bg_branch_macs_dict["branch_macs"]
        )

        # ===== 每张图实际激活的参数量 =====
        # 注意：stored params 表示模型总共存了多少参数；
        # activated params 表示这张图实际用到了哪些模块。
        activated_params = roi_head_params

        if roi_count > 0:
            activated_params += roi_branch_params

        if bg_count > 0:
            activated_params += bg_branch_params


        # ===== 10. 保存图像 =====
        stem = img_path.stem

        save_tensor_image(img_tensor, visual_dir / f"{stem}_input.png")
        save_tensor_image(recon_full, visual_dir / f"{stem}_recon_dual.png")

        diff = torch.abs(img_tensor - recon_full)
        diff = diff / (diff.max() + 1e-8)
        save_tensor_image(diff, visual_dir / f"{stem}_diff.png")

        compare = torch.cat(
            [img_tensor.detach().cpu(), recon_full.detach().cpu(), diff.detach().cpu()],
            dim=3
        )
        save_tensor_image(compare, visual_dir / f"{stem}_compare_input_recon_diff.png")

        save_roi_grid_big(grid, visual_dir / f"{stem}_pred_roi_grid_1024x896.png")
        save_overlay_roi(img_pil, grid, visual_dir / f"{stem}_overlay_pred_roi.png")

        # 保存 7×8 概率和值，方便后续查
        np.save(output_dir / f"{stem}_pred_grid.npy", grid)
        np.save(output_dir / f"{stem}_pred_prob.npy", probs.cpu().numpy())

        rows.append({
            "image": img_path.name,
            "roi_blocks": roi_count,
            "bg_blocks": bg_count,
            "full_psnr": full_psnr,
            "roi_psnr": roi_psnr if roi_psnr is not None else "",
            "bg_psnr": bg_psnr if bg_psnr is not None else "",
            "total_cbr": total_cbr,
            "compression_ratio_1_over_cbr": compression_ratio,
            "roi_feature_numel": roi_feature_numel,
            "bg_feature_numel": bg_feature_numel,
            "total_feature_numel": total_feature_numel,
            "threshold": threshold,
            "SNR_roi": SNR_roi,
            "SNR_bg": SNR_bg,

            "roi_head_params_M": to_million(roi_head_params),
            "roi_branch_params_M": to_million(roi_branch_params),
            "bg_branch_params_M": to_million(bg_branch_params),
            "total_stored_params_M": to_million(total_stored_params),
            "activated_params_M": to_million(activated_params),

            "roi_head_macs_G": to_giga(roi_head_macs),
            "roi_branch_macs_per_block_G": to_giga(roi_branch_macs_dict["branch_macs"]),
            "bg_branch_macs_per_block_G": to_giga(bg_branch_macs_dict["branch_macs"]),
            "effective_macs_G": to_giga(effective_macs),

            "roi_head_time_ms": roi_head_time_ms,
            "routing_time_ms": routing_time_ms,
            "roi_branch_time_ms": roi_branch_time_ms,
            "bg_branch_time_ms": bg_branch_time_ms,
            "stitching_time_ms": stitching_time_ms,
            "total_infer_time_ms": total_infer_time_ms,
        })

    # ===== 11. 保存 CSV =====
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "roi_blocks",
                "bg_blocks",
                "full_psnr",
                "roi_psnr",
                "bg_psnr",
                "total_cbr",
                "compression_ratio_1_over_cbr",
                "roi_feature_numel",
                "bg_feature_numel",
                "total_feature_numel",
                "threshold",
                "SNR_roi",
                "SNR_bg",

                "roi_head_params_M",
                "roi_branch_params_M",
                "bg_branch_params_M",
                "total_stored_params_M",
                "activated_params_M",

                "roi_head_macs_G",
                "roi_branch_macs_per_block_G",
                "bg_branch_macs_per_block_G",
                "effective_macs_G",

                "roi_head_time_ms",
                "routing_time_ms",
                "roi_branch_time_ms",
                "bg_branch_time_ms",
                "stitching_time_ms",
                "total_infer_time_ms",
            ]
        )
        writer.writeheader()
        writer.writerows(rows)

    if len(rows) > 0:
        avg_full_psnr = np.mean([r["full_psnr"] for r in rows])
        avg_cbr = np.mean([r["total_cbr"] for r in rows])
        avg_cr = np.mean([r["compression_ratio_1_over_cbr"] for r in rows])
        avg_roi_blocks = np.mean([r["roi_blocks"] for r in rows])
        avg_bg_blocks = np.mean([r["bg_blocks"] for r in rows])

        print("\n========== ROI Dual Full Inference Summary ==========")
        print(f"avg full PSNR = {avg_full_psnr:.4f}")
        print(f"avg total CBR = {avg_cbr:.6f}")
        print(f"avg compression ratio = {avg_cr:.4f}")
        print(f"avg ROI blocks = {avg_roi_blocks:.2f} / 56")
        print(f"avg BG blocks = {avg_bg_blocks:.2f} / 56")
        print(f"metrics saved to: {csv_path}")
        print(f"visual results saved to: {visual_dir}")

        avg_effective_macs = np.mean([r["effective_macs_G"] for r in rows])
        avg_total_time = np.mean([r["total_infer_time_ms"] for r in rows])
        avg_roi_head_time = np.mean([r["roi_head_time_ms"] for r in rows])
        avg_roi_branch_time = np.mean([r["roi_branch_time_ms"] for r in rows])
        avg_bg_branch_time = np.mean([r["bg_branch_time_ms"] for r in rows])

        print(f"total stored params = {to_million(total_stored_params):.4f} M")
        print(f"avg effective MACs = {avg_effective_macs:.4f} G")
        print(f"avg total inference delay = {avg_total_time:.4f} ms")
        print(f"avg ROI head time = {avg_roi_head_time:.4f} ms")
        print(f"avg ROI branch time = {avg_roi_branch_time:.4f} ms")
        print(f"avg BG branch time = {avg_bg_branch_time:.4f} ms")


import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# 加入项目根目录，避免在 tools/ 下运行时找不到 configs、utils、run 等模块
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import get_config
from utils.utils import seed_torch
from run.infer_roi_dual_full import infer_roi_dual_full

def parse_snr_list(s: str):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_modes(s: str):
    modes = [x.strip() for x in s.split(",") if x.strip()]
    valid = {"all_roi", "all_bg", "pred"}
    for m in modes:
        if m not in valid:
            raise ValueError(f"Unknown route mode: {m}. Valid modes: {valid}")
    return modes


def safe_float(x):
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def mean_col(rows, name):
    vals = []
    for r in rows:
        v = safe_float(r.get(name))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else ""


def summarize_metrics(csv_path: Path):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    if len(rows) == 0:
        raise RuntimeError(f"No rows found in {csv_path}")

    return {
        "num_images": len(rows),
        "avg_full_psnr": mean_col(rows, "full_psnr"),
        "avg_roi_psnr": mean_col(rows, "roi_psnr"),
        "avg_roi_msssim": mean_col(rows, "roi_msssim"),
        "avg_roi_lpips": mean_col(rows, "roi_lpips"),
        "avg_bg_psnr": mean_col(rows, "bg_psnr"),
        "avg_cbr": mean_col(rows, "total_cbr"),
        "avg_cr": mean_col(rows, "compression_ratio_1_over_cbr"),
        "avg_route_roi_blocks": mean_col(rows, "roi_blocks"),
        "avg_route_bg_blocks": mean_col(rows, "bg_blocks"),
        "avg_pred_roi_blocks": mean_col(rows, "pred_roi_blocks"),
        "avg_pred_bg_blocks": mean_col(rows, "pred_bg_blocks"),
        "total_stored_params_M": mean_col(rows, "total_stored_params_M"),
        "avg_effective_macs_G": mean_col(rows, "effective_macs_G"),
        "avg_total_time_ms": mean_col(rows, "total_infer_time_ms"),
        "avg_roi_head_time_ms": mean_col(rows, "roi_head_time_ms"),
        "avg_roi_branch_time_ms": mean_col(rows, "roi_branch_time_ms"),
        "avg_bg_branch_time_ms": mean_col(rows, "bg_branch_time_ms"),
    }


def write_summary(summary_rows, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "route_mode",
        "snr",
        "bg_low_size",
        "num_images",
        "avg_full_psnr",
        "avg_roi_psnr",
        "avg_roi_msssim",
        "avg_roi_lpips",
        "avg_bg_psnr",
        "avg_cbr",
        "avg_cr",
        "avg_route_roi_blocks",
        "avg_route_bg_blocks",
        "avg_pred_roi_blocks",
        "avg_pred_bg_blocks",
        "total_stored_params_M",
        "avg_effective_macs_G",
        "avg_total_time_ms",
        "avg_roi_head_time_ms",
        "avg_roi_branch_time_ms",
        "avg_bg_branch_time_ms",
        "output_dir",
    ]

    with open(save_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-path",
        type=str,
        default="/root/autodl-tmp/mambajscc/MambaJSCC",
        help="MambaJSCC project root path",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/root/autodl-tmp/datasets/MUSeg/test_official/Image_1024x896",
    )
    parser.add_argument(
        "--coarse-roi-ckpt",
        type=str,
        default="/root/autodl-tmp/mambajscc/MambaJSCC/checkpoints/coarse_roi/best_coarse_roi_finetune.pth",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="/root/autodl-tmp/mambajscc/MambaJSCC/outputs/awgn_snr_routes",
    )
    parser.add_argument(
        "--snrs",
        type=str,
        default="1,3,5,7,9,11,13,15,17,20",
        help="Comma-separated SNR list",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="all_roi,all_bg,pred",
        help="Comma-separated route modes: all_roi, all_bg, pred",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--image-start", type=int, default=400)
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--bg-low-size", type=int, default=64)
    parser.add_argument("--save-visual", action="store_true")

    args = parser.parse_args()

    seed_torch()

    project_path = Path(args.project_path)

    class RoiArgs:
        model_config_path = str(project_path / "configs/vssm/vssm_tiny_MUSeg.yaml")
        train_config_path = str(project_path / "configs/train/vssm_tiny_MUSeg_rayleigh_seed1024.yaml")

    class BgArgs:
        model_config_path = str(project_path / "configs/vssm/vssm_tiny_MUSeg_bg.yaml")
        train_config_path = str(project_path / "configs/train/vssm_tiny_MUSeg_rayleigh_seed1024.yaml")

    roi_config = get_config(RoiArgs)
    bg_config = get_config(BgArgs)

    print(f"[Channel] ROI channel = {roi_config.CHANNEL.TYPE}")
    print(f"[Channel] BG channel  = {bg_config.CHANNEL.TYPE}")

    snr_list = parse_snr_list(args.snrs)
    route_modes = parse_modes(args.modes)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for snr in snr_list:
        for mode in route_modes:
            snr_name = str(snr).replace(".", "p")
            out_dir = output_root / f"{mode}_snr{snr_name}_bg{args.bg_low_size}"

            print("\n" + "=" * 80)
            print(f"[Experiment] route_mode = {mode}, SNR = {snr}, bg_low_size = {args.bg_low_size}")
            print(f"[Output] {out_dir}")
            print("=" * 80)

            infer_roi_dual_full(
                roi_config=roi_config,
                bg_config=bg_config,
                image_dir=args.image_dir,
                coarse_roi_ckpt=args.coarse_roi_ckpt,
                output_dir=str(out_dir),
                threshold=args.threshold,
                image_start=args.image_start,
                max_images=args.max_images,
                bg_low_size=args.bg_low_size,
                route_mode=mode,
                snr_override=snr,
                save_visual=args.save_visual,
        
                profile_complexity=False,
            )

            metrics_csv = out_dir / "metrics.csv"
            summary = summarize_metrics(metrics_csv)
            summary.update({
                "route_mode": mode,
                "snr": snr,
                "bg_low_size": args.bg_low_size,
                "output_dir": str(out_dir),
            })
            summary_rows.append(summary)

            write_summary(summary_rows, output_root / "summary.csv")

    print("\n========== Experiment Finished ==========")
    print(f"summary saved to: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
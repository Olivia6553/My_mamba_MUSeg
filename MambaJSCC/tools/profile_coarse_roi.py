import csv
import time
import sys
from pathlib import Path

import torch
from thop import profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.coarse_roi_net import LightCoarseROINet


CKPT_PATH = ROOT / "checkpoints" / "coarse_roi" / "best_coarse_roi_finetune.pth"
SAVE_DIR = ROOT / "outputs" / "coarse_roi_eval"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 1
C = 3
H = 896
W = 1024

WARMUP = 30
REPEAT = 100


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device =", device)

    model = LightCoarseROINet(base_ch=16).to(device)

    if CKPT_PATH.exists():
        state_dict = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"loaded ckpt: {CKPT_PATH}")
    else:
        print("[warning] checkpoint not found, only profile model structure")

    model.eval()

    x = torch.randn(BATCH_SIZE, C, H, W).to(device)

    params = count_params(model)
    macs, _ = profile(model, inputs=(x,), verbose=False)

    # 预热
    for _ in range(WARMUP):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(REPEAT):
        _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    avg_time_ms = (end - start) / REPEAT * 1000

    params_m = params / 1e6
    macs_g = macs / 1e9

    print("\n========== Coarse ROI Head Complexity ==========")
    print(f"Params      = {params_m:.6f} M")
    print(f"MACs        = {macs_g:.6f} G")
    print(f"Avg time    = {avg_time_ms:.4f} ms / image")
    print(f"Input size  = [{BATCH_SIZE}, {C}, {H}, {W}]")
    print(f"Output grid = [1, 1, 7, 8]")

    csv_path = SAVE_DIR / "coarse_roi_complexity.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["params_M", params_m])
        writer.writerow(["macs_G", macs_g])
        writer.writerow(["avg_time_ms_per_image", avg_time_ms])
        writer.writerow(["input_size", f"{BATCH_SIZE}x{C}x{H}x{W}"])
        writer.writerow(["output_grid", "1x1x7x8"])

    print(f"\ncomplexity saved to: {csv_path}")


if __name__ == "__main__":
    main()
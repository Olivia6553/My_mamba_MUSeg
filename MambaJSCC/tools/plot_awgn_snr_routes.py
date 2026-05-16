import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        default="/root/autodl-tmp/mambajscc/MambaJSCC/outputs/awgn_snr_routes/summary.csv",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/root/autodl-tmp/mambajscc/MambaJSCC/outputs/awgn_snr_routes/figures",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    df = df.sort_values(["route_mode", "snr"])

    label_map = {
        "all_roi": "All ROI branch",
        "all_bg": "All BG branch",
        "pred": "ROI/BG dual branch",
    }

    # 1. Full PSNR - SNR
    plt.figure(figsize=(7, 5))
    for mode in ["all_roi", "pred", "all_bg"]:
        sub = df[df["route_mode"] == mode]
        if len(sub) == 0:
            continue
        plt.plot(
            sub["snr"],
            sub["avg_full_psnr"],
            marker="o",
            label=label_map.get(mode, mode),
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("Full PSNR (dB)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "psnr_snr_full.png", dpi=300)
    plt.close()

    # 2. ROI PSNR - SNR
    plt.figure(figsize=(7, 5))
    for mode in ["all_roi", "pred", "all_bg"]:
        sub = df[df["route_mode"] == mode]
        if len(sub) == 0 or "avg_roi_psnr" not in sub:
            continue
        plt.plot(
            sub["snr"],
            sub["avg_roi_psnr"],
            marker="o",
            label=label_map.get(mode, mode),
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel("ROI PSNR (dB)")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "psnr_snr_roi.png", dpi=300)
    plt.close()

    # 3. CBR bar, use average over SNR
    cbr_df = df.groupby("route_mode", as_index=False)["avg_cbr"].mean()
    cbr_df["label"] = cbr_df["route_mode"].map(label_map).fillna(cbr_df["route_mode"])

    plt.figure(figsize=(6, 5))
    plt.bar(cbr_df["label"], cbr_df["avg_cbr"])
    plt.ylabel("Average CBR")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(save_dir / "avg_cbr_bar.png", dpi=300)
    plt.close()

    print(f"figures saved to: {save_dir}")


if __name__ == "__main__":
    main()
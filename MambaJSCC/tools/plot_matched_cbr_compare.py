
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 路径配置
# =========================
# 你的 ROI/BG 方法 SNR=9 结果
your_snr9_csv = Path("outputs/awgn_routes_full_snr9/summary.csv")

# OUTCHANS20 同 CBR 基线 SNR=9 结果
base_snr9_csv = Path("outputs/baseline_mambajscc_out20_snr9/summary.csv")

# 可选：你的 ROI/BG 方法 full10 结果
your_full10_csv = Path("outputs/awgn_snr_routes_full10/summary.csv")

# 可选：OUTCHANS20 同 CBR 基线 full10 结果
base_full10_csv = Path("outputs/baseline_mambajscc_out20_full10/summary.csv")

save_dir = Path("outputs/matched_cbr_compare_figures")
save_dir.mkdir(parents=True, exist_ok=True)


def read_csv_checked(path):
    if not path.exists():
        print(f"[Skip] Not found: {path}")
        return None
    return pd.read_csv(path)


def get_row(df, mode):
    sub = df[df["route_mode"] == mode]
    if len(sub) == 0:
        raise ValueError(f"route_mode={mode} not found")
    return sub.iloc[0]


# =========================
# 1. SNR=9 单点对比
# =========================
your_snr9 = read_csv_checked(your_snr9_csv)
base_snr9 = read_csv_checked(base_snr9_csv)

if your_snr9 is not None and base_snr9 is not None:
    high = get_row(your_snr9, "all_roi")
    low = get_row(your_snr9, "all_bg")
    yours = get_row(your_snr9, "pred")
    base = get_row(base_snr9, "all_roi")

    rows = [
        {
            "method": "MambaJSCC-High",
            "Full PSNR": high["avg_full_psnr"],
            "ROI PSNR": high["avg_roi_psnr"],
            "BG PSNR": high["avg_bg_psnr"],
            "CBR": high["avg_cbr"],
            "CR": high["avg_cr"],
        },
        {
            "method": "MambaJSCC-Matched",
            "Full PSNR": base["avg_full_psnr"],
            "ROI PSNR": base["avg_roi_psnr"],
            "BG PSNR": base["avg_bg_psnr"],
            "CBR": base["avg_cbr"],
            "CR": base["avg_cr"],
        },
        {
            "method": "ROI/BG-MambaJSCC",
            "Full PSNR": yours["avg_full_psnr"],
            "ROI PSNR": yours["avg_roi_psnr"],
            "BG PSNR": yours["avg_bg_psnr"],
            "CBR": yours["avg_cbr"],
            "CR": yours["avg_cr"],
        },
        {
            "method": "MambaJSCC-Low",
            "Full PSNR": low["avg_full_psnr"],
            "ROI PSNR": low["avg_roi_psnr"],
            "BG PSNR": low["avg_bg_psnr"],
            "CBR": low["avg_cbr"],
            "CR": low["avg_cr"],
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(save_dir / "matched_cbr_snr9_table.csv", index=False)

    print("\n========== SNR=9 Comparison ==========")
    print(df.to_string(index=False))

    # 1.1 PSNR 柱状图
    x = list(range(len(df)))
    width = 0.25

    plt.figure(figsize=(9, 5))
    plt.bar([i - width for i in x], df["Full PSNR"], width=width, label="Full PSNR")
    plt.bar(x, df["ROI PSNR"], width=width, label="ROI PSNR")
    plt.bar([i + width for i in x], df["BG PSNR"], width=width, label="BG PSNR")

    plt.xticks(x, df["method"], rotation=20, ha="right")
    plt.ylabel("PSNR (dB)")
    plt.title("Matched CBR Comparison at AWGN SNR = 9 dB")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "snr9_psnr_comparison.png", dpi=300)
    plt.close()

    # 1.2 CBR 柱状图
    plt.figure(figsize=(8, 5))
    plt.bar(df["method"], df["CBR"])
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Average CBR")
    plt.title("Average CBR Comparison")
    plt.tight_layout()
    plt.savefig(save_dir / "snr9_cbr_comparison.png", dpi=300)
    plt.close()

    # 1.3 ROI PSNR - CBR 散点图
    plt.figure(figsize=(7, 5))
    plt.scatter(df["CBR"], df["ROI PSNR"], s=70)

    for _, row in df.iterrows():
        plt.annotate(
            row["method"],
            (row["CBR"], row["ROI PSNR"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )

    plt.xlabel("Average CBR")
    plt.ylabel("ROI PSNR (dB)")
    plt.title("ROI PSNR-CBR Tradeoff at AWGN SNR = 9 dB")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(save_dir / "snr9_roi_psnr_cbr_tradeoff.png", dpi=300)
    plt.close()

    print(f"\n[SNR=9 figures saved to] {save_dir}")


# =========================
# 2. Full10 PSNR-SNR 曲线对比
# =========================
your_full10 = read_csv_checked(your_full10_csv)
base_full10 = read_csv_checked(base_full10_csv)

if your_full10 is not None and base_full10 is not None:
    high_curve = your_full10[your_full10["route_mode"] == "all_roi"].copy()
    low_curve = your_full10[your_full10["route_mode"] == "all_bg"].copy()
    your_curve = your_full10[your_full10["route_mode"] == "pred"].copy()
    base_curve = base_full10[base_full10["route_mode"] == "all_roi"].copy()

    curves = [
        ("MambaJSCC-High", high_curve),
        ("MambaJSCC-Matched", base_curve),
        ("ROI/BG-MambaJSCC", your_curve),
        ("MambaJSCC-Low", low_curve),
    ]

    # 2.1 Full PSNR-SNR
    plt.figure(figsize=(7, 5))
    for name, sub in curves:
        sub = sub.sort_values("snr")
        plt.plot(sub["snr"], sub["avg_full_psnr"], marker="o", label=name)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Full PSNR (dB)")
    plt.title("Full PSNR versus SNR under AWGN")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "full_psnr_snr_matched_cbr.png", dpi=300)
    plt.close()

    # 2.2 ROI PSNR-SNR
    plt.figure(figsize=(7, 5))
    for name, sub in curves:
        sub = sub.sort_values("snr")
        plt.plot(sub["snr"], sub["avg_roi_psnr"], marker="o", label=name)

    plt.xlabel("SNR (dB)")
    plt.ylabel("ROI PSNR (dB)")
    plt.title("ROI PSNR versus SNR under AWGN")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "roi_psnr_snr_matched_cbr.png", dpi=300)
    plt.close()

    print(f"\n[Full10 curve figures saved to] {save_dir}")


print("\nDone.")

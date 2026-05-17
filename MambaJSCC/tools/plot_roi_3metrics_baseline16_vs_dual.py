
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 路径设置
# =========================
baseline_csv = Path("outputs/baseline_mambajscc_out16_full10_roi3m/summary.csv")
dual_csv = Path("outputs/roi_bg_pred_full10_roi3m/summary.csv")

save_dir = Path("outputs/roi_3metrics_baseline16_vs_dual")
save_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 读取数据
# =========================
baseline = pd.read_csv(baseline_csv)
dual = pd.read_csv(dual_csv)

# 基线 OUTCHANS16 使用 all_roi
baseline = baseline[baseline["route_mode"] == "all_roi"].copy()

# 双分支模型使用 pred
dual = dual[dual["route_mode"] == "pred"].copy()

baseline = baseline.sort_values("snr")
dual = dual.sort_values("snr")


# =========================
# 检查必要字段
# =========================
required_cols = [
    "snr",
    "avg_cbr",
    "avg_roi_psnr",
    "avg_roi_msssim",
    "avg_roi_lpips",
]

for col in required_cols:
    if col not in baseline.columns:
        raise ValueError(f"{col} not found in baseline csv: {baseline_csv}")
    if col not in dual.columns:
        raise ValueError(f"{col} not found in dual csv: {dual_csv}")


# =========================
# 保存对比表
# =========================
table = pd.DataFrame({
    "snr": baseline["snr"].values,

    "baseline_cbr": baseline["avg_cbr"].values,
    "dual_cbr": dual["avg_cbr"].values,

    "baseline_roi_psnr": baseline["avg_roi_psnr"].values,
    "dual_roi_psnr": dual["avg_roi_psnr"].values,

    "baseline_roi_msssim": baseline["avg_roi_msssim"].values,
    "dual_roi_msssim": dual["avg_roi_msssim"].values,

    "baseline_roi_lpips": baseline["avg_roi_lpips"].values,
    "dual_roi_lpips": dual["avg_roi_lpips"].values,
})

table["psnr_gain"] = table["dual_roi_psnr"] - table["baseline_roi_psnr"]
table["msssim_gain"] = table["dual_roi_msssim"] - table["baseline_roi_msssim"]

# LPIPS 越低越好，所以这里是 baseline - dual，正值表示你的模型更好
table["lpips_reduction"] = table["baseline_roi_lpips"] - table["dual_roi_lpips"]

table_path = save_dir / "roi_3metrics_compare_table.csv"
table.to_csv(table_path, index=False)

print("\n========== ROI 3 Metrics Comparison ==========")
print(table.to_string(index=False))
print(f"\nComparison table saved to: {table_path}")


# =========================
# 统一画图函数
# =========================
def plot_metric(
    y_col,
    ylabel,
    title,
    filename,
    lpips=False,
):
    plt.figure(figsize=(7, 5))

    plt.plot(
        baseline["snr"],
        baseline[y_col],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="MambaJSCC baseline, CBR=0.0104"
    )

    plt.plot(
        dual["snr"],
        dual[y_col],
        marker="s",
        linewidth=1.8,
        markersize=6,
        label="ROI/BG-MambaJSCC"
    )

    plt.xlabel("SNR (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()

    plt.savefig(save_dir / filename, dpi=300)
    plt.close()


# =========================
# 1. ROI PSNR-SNR
# =========================
plot_metric(
    y_col="avg_roi_psnr",
    ylabel="ROI PSNR (dB)",
    title="ROI PSNR versus SNR under AWGN",
    filename="roi_psnr_snr_baseline16_vs_dual.png",
)

# =========================
# 2. ROI MS-SSIM-SNR
# =========================
plot_metric(
    y_col="avg_roi_msssim",
    ylabel="ROI MS-SSIM",
    title="ROI MS-SSIM versus SNR under AWGN",
    filename="roi_msssim_snr_baseline16_vs_dual.png",
)

# =========================
# 3. ROI LPIPS-SNR
# =========================
plot_metric(
    y_col="avg_roi_lpips",
    ylabel="ROI LPIPS",
    title="ROI LPIPS versus SNR under AWGN",
    filename="roi_lpips_snr_baseline16_vs_dual.png",
)

print(f"\nFigures saved to: {save_dir}")

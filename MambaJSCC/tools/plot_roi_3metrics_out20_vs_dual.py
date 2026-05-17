
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 文件路径
# =========================
baseline_csv = Path("outputs/baseline_mambajscc_out20_full10_roi3m/summary.csv")
dual_csv = Path("outputs/roi_bg_pred_full10_roi3m/summary.csv")

save_dir = Path("outputs/fig_out20_vs_dual_roi_3metrics")
save_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 读取数据
# =========================
baseline = pd.read_csv(baseline_csv)
dual = pd.read_csv(dual_csv)

# OUTCHANS20 基线用 all_roi
baseline = baseline[baseline["route_mode"] == "all_roi"].copy()

# 双分支模型用 pred
dual = dual[dual["route_mode"] == "pred"].copy()

baseline = baseline.sort_values("snr")
dual = dual.sort_values("snr")


# =========================
# 字段检查
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
# 保存对比数据表
# =========================
table = pd.DataFrame({
    "snr": baseline["snr"].values,

    "out20_cbr": baseline["avg_cbr"].values,
    "dual_cbr": dual["avg_cbr"].values,

    "out20_roi_psnr": baseline["avg_roi_psnr"].values,
    "dual_roi_psnr": dual["avg_roi_psnr"].values,

    "out20_roi_msssim": baseline["avg_roi_msssim"].values,
    "dual_roi_msssim": dual["avg_roi_msssim"].values,

    "out20_roi_lpips": baseline["avg_roi_lpips"].values,
    "dual_roi_lpips": dual["avg_roi_lpips"].values,
})

table["psnr_gain"] = table["dual_roi_psnr"] - table["out20_roi_psnr"]
table["msssim_gain"] = table["dual_roi_msssim"] - table["out20_roi_msssim"]

# LPIPS 越低越好，所以 out20 - dual 为正表示双分支更好
table["lpips_reduction"] = table["out20_roi_lpips"] - table["dual_roi_lpips"]

table_path = save_dir / "out20_vs_dual_roi_3metrics_table.csv"
table.to_csv(table_path, index=False)

print("\n========== OUTCHANS20 vs ROI/BG-MambaJSCC ==========")
print(table.to_string(index=False))
print(f"\nComparison table saved to: {table_path}")


# =========================
# 画图函数
# =========================
def plot_metric(y_col, ylabel, title, filename):
    plt.figure(figsize=(7, 5))

    plt.plot(
        baseline["snr"],
        baseline[y_col],
        marker="o",
        linewidth=1.8,
        markersize=6,
        label="MambaJSCC-OUTCHANS20, CBR≈0.0130"
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
# 1. ROI PSNR
# =========================
plot_metric(
    y_col="avg_roi_psnr",
    ylabel="ROI PSNR (dB)",
    title="ROI PSNR versus SNR under AWGN",
    filename="roi_psnr_snr_out20_vs_dual.png",
)

# =========================
# 2. ROI MS-SSIM
# =========================
plot_metric(
    y_col="avg_roi_msssim",
    ylabel="ROI MS-SSIM",
    title="ROI MS-SSIM versus SNR under AWGN",
    filename="roi_msssim_snr_out20_vs_dual.png",
)

# =========================
# 3. ROI LPIPS
# =========================
plot_metric(
    y_col="avg_roi_lpips",
    ylabel="ROI LPIPS",
    title="ROI LPIPS versus SNR under AWGN",
    filename="roi_lpips_snr_out20_vs_dual.png",
)

print(f"\nFigures saved to: {save_dir}")


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# =========================
# 输入文件
# =========================
out16_csv = Path("outputs/rayleigh_baseline_mambajscc_out16_full10_roi3m/summary.csv")
out20_csv = Path("outputs/rayleigh_baseline_mambajscc_out20_full10_roi3m/summary.csv")
dual_csv  = Path("outputs/rayleigh_roi_bg_pred_full10_roi3m/summary.csv")

save_dir = Path("outputs/fig_rayleigh_6figs")
save_dir.mkdir(parents=True, exist_ok=True)


# =========================
# 读数据
# =========================
if not out16_csv.exists():
    raise FileNotFoundError(f"Missing file: {out16_csv}")
if not out20_csv.exists():
    raise FileNotFoundError(f"Missing file: {out20_csv}")
if not dual_csv.exists():
    raise FileNotFoundError(f"Missing file: {dual_csv}")

out16 = pd.read_csv(out16_csv)
out20 = pd.read_csv(out20_csv)
dual = pd.read_csv(dual_csv)

out16 = out16[out16["route_mode"] == "all_roi"].copy().sort_values("snr")
out20 = out20[out20["route_mode"] == "all_roi"].copy().sort_values("snr")
dual  = dual[dual["route_mode"]  == "pred"].copy().sort_values("snr")


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

for name, df in [("out16", out16), ("out20", out20), ("dual", dual)]:
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} not found in {name} summary.csv")


# =========================
# 保存对比表
# =========================
table16 = pd.DataFrame({
    "snr": out16["snr"].values,
    "out16_cbr": out16["avg_cbr"].values,
    "dual_cbr": dual["avg_cbr"].values,

    "out16_roi_psnr": out16["avg_roi_psnr"].values,
    "dual_roi_psnr": dual["avg_roi_psnr"].values,

    "out16_roi_msssim": out16["avg_roi_msssim"].values,
    "dual_roi_msssim": dual["avg_roi_msssim"].values,

    "out16_roi_lpips": out16["avg_roi_lpips"].values,
    "dual_roi_lpips": dual["avg_roi_lpips"].values,
})

table16["psnr_gain"] = table16["dual_roi_psnr"] - table16["out16_roi_psnr"]
table16["msssim_gain"] = table16["dual_roi_msssim"] - table16["out16_roi_msssim"]
table16["lpips_reduction"] = table16["out16_roi_lpips"] - table16["dual_roi_lpips"]

table20 = pd.DataFrame({
    "snr": out20["snr"].values,
    "out20_cbr": out20["avg_cbr"].values,
    "dual_cbr": dual["avg_cbr"].values,

    "out20_roi_psnr": out20["avg_roi_psnr"].values,
    "dual_roi_psnr": dual["avg_roi_psnr"].values,

    "out20_roi_msssim": out20["avg_roi_msssim"].values,
    "dual_roi_msssim": dual["avg_roi_msssim"].values,

    "out20_roi_lpips": out20["avg_roi_lpips"].values,
    "dual_roi_lpips": dual["avg_roi_lpips"].values,
})

table20["psnr_gain"] = table20["dual_roi_psnr"] - table20["out20_roi_psnr"]
table20["msssim_gain"] = table20["dual_roi_msssim"] - table20["out20_roi_msssim"]
table20["lpips_reduction"] = table20["out20_roi_lpips"] - table20["dual_roi_lpips"]

table16.to_csv(save_dir / "rayleigh_out16_vs_dual_table.csv", index=False)
table20.to_csv(save_dir / "rayleigh_out20_vs_dual_table.csv", index=False)

print("\n========== Rayleigh: OUT16 vs Dual ==========")
print(table16.to_string(index=False))
print("\n========== Rayleigh: OUT20 vs Dual ==========")
print(table20.to_string(index=False))


# =========================
# 通用画图函数
# =========================
def plot_two_curves(
    x1, y1, label1,
    x2, y2, label2,
    xlabel, ylabel, title, filename
):
    plt.figure(figsize=(7, 5))

    plt.plot(
        x1, y1,
        marker="o",
        linewidth=1.8,
        markersize=6,
        label=label1
    )

    plt.plot(
        x2, y2,
        marker="s",
        linewidth=1.8,
        markersize=6,
        label=label2
    )

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / filename, dpi=300)
    plt.close()


# =========================
# 1~3: OUT16 vs Dual
# =========================
plot_two_curves(
    out16["snr"], out16["avg_roi_psnr"], "MambaJSCC-OUTCHANS16, CBR≈0.0104",
    dual["snr"],  dual["avg_roi_psnr"],  "ROI/BG-MambaJSCC",
    "SNR (dB)", "ROI PSNR (dB)",
    "ROI PSNR versus SNR under Rayleigh (OUT16 vs Dual)",
    "rayleigh_roi_psnr_snr_out16_vs_dual.png"
)

plot_two_curves(
    out16["snr"], out16["avg_roi_msssim"], "MambaJSCC-OUTCHANS16, CBR≈0.0104",
    dual["snr"],  dual["avg_roi_msssim"],  "ROI/BG-MambaJSCC",
    "SNR (dB)", "ROI MS-SSIM",
    "ROI MS-SSIM versus SNR under Rayleigh (OUT16 vs Dual)",
    "rayleigh_roi_msssim_snr_out16_vs_dual.png"
)

plot_two_curves(
    out16["snr"], out16["avg_roi_lpips"], "MambaJSCC-OUTCHANS16, CBR≈0.0104",
    dual["snr"],  dual["avg_roi_lpips"],  "ROI/BG-MambaJSCC",
    "SNR (dB)", "ROI LPIPS",
    "ROI LPIPS versus SNR under Rayleigh (OUT16 vs Dual)",
    "rayleigh_roi_lpips_snr_out16_vs_dual.png"
)


# =========================
# 4~6: OUT20 vs Dual
# =========================
plot_two_curves(
    out20["snr"], out20["avg_roi_psnr"], "MambaJSCC-OUTCHANS20, CBR≈0.0130",
    dual["snr"],  dual["avg_roi_psnr"],  "ROI/BG-MambaJSCC",
    "SNR (dB)", "ROI PSNR (dB)",
    "ROI PSNR versus SNR under Rayleigh (OUT20 vs Dual)",
    "rayleigh_roi_psnr_snr_out20_vs_dual.png"
)

plot_two_curves(
    out20["snr"], out20["avg_roi_msssim"], "MambaJSCC-OUTCHANS20, CBR≈0.0130",
    dual["snr"],  dual["avg_roi_msssim"],  "ROI/BG-MambaJSCC",
    "SNR (dB)", "ROI MS-SSIM",
    "ROI MS-SSIM versus SNR under Rayleigh (OUT20 vs Dual)",
    "rayleigh_roi_msssim_snr_out20_vs_dual.png"
)

plot_two_curves(
    out20["snr"], out20["avg_roi_lpips"], "MambaJSCC-OUTCHANS20, CBR≈0.0130",
    dual["snr"],  dual["avg_roi_lpips"],  "ROI/BG-MambaJSCC",
    "SNR (dB)", "ROI LPIPS",
    "ROI LPIPS versus SNR under Rayleigh (OUT20 vs Dual)",
    "rayleigh_roi_lpips_snr_out20_vs_dual.png"
)

print(f"\nFigures saved to: {save_dir}")

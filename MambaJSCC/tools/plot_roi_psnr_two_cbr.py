
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 这里改成你的两个 metrics.csv 路径
# =========================
high_cbr_csv = Path("outputs/awgn_routes_full_snr9/all_roi_snr9p0_bg64/metrics.csv")
low_cbr_csv = Path("outputs/baseline_mambajscc_out20_snr9/all_roi_snr9p0_bg64/metrics.csv")

save_dir = Path("outputs/roi_psnr_two_cbr_compare")
save_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 读取数据
# =========================
high = pd.read_csv(high_cbr_csv)
low = pd.read_csv(low_cbr_csv)

# 只保留有效 roi_psnr
high_roi = high["roi_psnr"].dropna().astype(float)
low_roi = low["roi_psnr"].dropna().astype(float)

# 标签
labels = [
    "CBR=0.02083\nOUTCHANS32",
    "CBR=0.01302\nOUTCHANS20"
]

# =========================
# 打印统计量
# =========================
stat = pd.DataFrame([
    {
        "setting": "CBR=0.02083 OUTCHANS32",
        "num_images": len(high_roi),
        "mean_roi_psnr": high_roi.mean(),
        "median_roi_psnr": high_roi.median(),
        "std_roi_psnr": high_roi.std(),
        "min_roi_psnr": high_roi.min(),
        "max_roi_psnr": high_roi.max(),
    },
    {
        "setting": "CBR=0.01302 OUTCHANS20",
        "num_images": len(low_roi),
        "mean_roi_psnr": low_roi.mean(),
        "median_roi_psnr": low_roi.median(),
        "std_roi_psnr": low_roi.std(),
        "min_roi_psnr": low_roi.min(),
        "max_roi_psnr": low_roi.max(),
    }
])

stat.to_csv(save_dir / "roi_psnr_two_cbr_stats.csv", index=False)
print(stat.to_string(index=False))

# =========================
# 1. 箱线图
# =========================
plt.figure(figsize=(6, 5))
plt.boxplot(
    [high_roi, low_roi],
    labels=labels,
    showmeans=True
)
plt.ylabel("ROI PSNR (dB)")
plt.title("ROI PSNR Distribution under Two CBR Settings")
plt.grid(True, linestyle="--", linewidth=0.5, axis="y")
plt.tight_layout()
plt.savefig(save_dir / "roi_psnr_boxplot_two_cbr.png", dpi=300)
plt.close()

# =========================
# 2. 每张图像的 ROI PSNR 对比散点图
# =========================
n = min(len(high_roi), len(low_roi))
x = list(range(n))

plt.figure(figsize=(9, 5))
plt.scatter(x, high_roi.iloc[:n], s=8, label="CBR=0.02083 OUTCHANS32", alpha=0.7)
plt.scatter(x, low_roi.iloc[:n], s=8, label="CBR=0.01302 OUTCHANS20", alpha=0.7)
plt.xlabel("Image index")
plt.ylabel("ROI PSNR (dB)")
plt.title("Per-image ROI PSNR under Two CBR Settings")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(save_dir / "roi_psnr_scatter_two_cbr.png", dpi=300)
plt.close()

# =========================
# 3. 差值图：高 CBR - 低 CBR
# =========================
diff = high_roi.iloc[:n].reset_index(drop=True) - low_roi.iloc[:n].reset_index(drop=True)

plt.figure(figsize=(9, 5))
plt.plot(x, diff, linewidth=1)
plt.axhline(0, linestyle="--", linewidth=1)
plt.xlabel("Image index")
plt.ylabel("ROI PSNR Difference (dB)")
plt.title("ROI PSNR Difference: CBR=0.02083 minus CBR=0.01302")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(save_dir / "roi_psnr_difference_two_cbr.png", dpi=300)
plt.close()

# =========================
# 4. CDF 曲线
# =========================
high_sorted = high_roi.sort_values().reset_index(drop=True)
low_sorted = low_roi.sort_values().reset_index(drop=True)

high_cdf = [(i + 1) / len(high_sorted) for i in range(len(high_sorted))]
low_cdf = [(i + 1) / len(low_sorted) for i in range(len(low_sorted))]

plt.figure(figsize=(7, 5))
plt.plot(high_sorted, high_cdf, label="CBR=0.02083 OUTCHANS32")
plt.plot(low_sorted, low_cdf, label="CBR=0.01302 OUTCHANS20")
plt.xlabel("ROI PSNR (dB)")
plt.ylabel("CDF")
plt.title("CDF of ROI PSNR under Two CBR Settings")
plt.legend()
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.savefig(save_dir / "roi_psnr_cdf_two_cbr.png", dpi=300)
plt.close()

print(f"\nFigures saved to: {save_dir}")

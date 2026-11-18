import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

CSV_FILE = "./data/jaccard_check.csv"
with open(CSV_FILE, encoding="utf-8") as f:
    K = int(re.search(r"k=(\d+)", f.readline()).group(1))

df = pd.read_csv(CSV_FILE, comment="#")
est = df["estimate_jaccard"].to_numpy()
J = df["exact_jaccard"].to_numpy()
err = est - J


# Độ lệch chuẩn lý thuyết: sigma_i = sqrt(J_i (1 - J_i) / k)
sigma_i = np.sqrt(J * (1 - J) / K)
sigma_th = np.mean(sigma_i)
#sigma_th = np.sqrt(np.mean(sigma_i ** 2))  # sigma lý thuyết gộp (RMS) cho cả mẫu
sigma_max = 1 / (2 * np.sqrt(K))           # cận trên 1/(2 sqrt(k))

# Thống kê thực nghiệm
print(f"Số cặp                      : {len(df)}")
print(f"Sai số trung bình (bias)    : {err.mean():+.5f}")
print(f"Độ lệch chuẩn thực nghiệm   : {err.std(ddof=1):.5f}")
print(f"Độ lệch chuẩn lý thuyết     : {sigma_th:.5f}  (cận trên {sigma_max:.5f})")
print(f"Sai số tuyệt đối lớn nhất   : {np.abs(err).max():.5f}")

# Sai số chuẩn hóa z = err / sigma_i (bỏ cặp có sigma = 0, tức J = 0 hoặc 1)
mask = sigma_i > 0
z = err[mask] / sigma_i[mask]
print(f"z: trung bình {z.mean():+.3f}, độ lệch chuẩn {z.std(ddof=1):.3f}  (kỳ vọng 0 và 1)")
print(f"Tỉ lệ |err| <= 1 sigma      : {np.mean(np.abs(z) <= 1):.3f}  (kỳ vọng ~0.683)")
print(f"Tỉ lệ |err| <= 2 sigma      : {np.mean(np.abs(z) <= 2):.3f}  (kỳ vọng ~0.954)")

# Hình 1: histogram sai số + đường chuẩn N(0, sigma_th)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
ax.hist(err, bins=40, density=True, alpha=0.7, edgecolor="black", label="Thực nghiệm")
xs = np.linspace(err.min(), err.max(), 400)
pdf = np.exp(-xs ** 2 / (2 * sigma_th ** 2)) / (sigma_th * np.sqrt(2 * np.pi))
ax.plot(xs, pdf, "r-", lw=2, label=f"N(0, σ={sigma_th:.4f})")
ax.axvline(0, color="k", ls="--", lw=1)
ax.set_xlabel("Sai số Ĵ - J")
ax.set_ylabel("Mật độ")
ax.set_title(f"Histogram sai số (k = {K}, n = {len(df)})")
ax.legend()

# Hình 2: histogram sai số chuẩn hóa so với N(0, 1)
ax = axes[1]
ax.hist(z, bins=40, density=True, alpha=0.7, edgecolor="black", label="Thực nghiệm")
zs = np.linspace(-4, 4, 400)
ax.plot(zs, np.exp(-zs ** 2 / 2) / np.sqrt(2 * np.pi), "r-", lw=2, label="N(0, 1)")
ax.set_xlabel("z = (Ĵ - J) / σ_i")
ax.set_title("Sai số chuẩn hóa")
ax.legend()

# Hình 3: Ĵ theo J, kèm dải ±2σ lý thuyết
ax = axes[2]
ax.scatter(J, est, s=8, alpha=0.5)
js = np.linspace(0, 1, 200)
band = 2 * np.sqrt(js * (1 - js) / K)
ax.plot(js, js, "r-", lw=1.5, label="Ĵ = J")
ax.plot(js, js + band, "g--", lw=1, label="±2σ lý thuyết")
ax.plot(js, js - band, "g--", lw=1)
ax.set_xlabel("J thật")
ax.set_ylabel("Ĵ ước lượng")
ax.set_title("Ĵ theo J")
ax.legend()

plt.tight_layout()
# plt.savefig("./data/error_histogram.png", dpi=150)   # bỏ comment nếu muốn lưu ảnh
plt.show()

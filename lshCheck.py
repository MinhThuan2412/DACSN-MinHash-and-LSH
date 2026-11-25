import csv
import json
import os
import time

import numpy as np

from minHash_LSH import (
    char_shingles, word_shingles, hash_shingles,
    MinHasher, lsh_candidates, exact_jaccard, estimate_jaccard,
    optimal_k, optimal_br, nearest_threshold_br,
)

IN_FILE = "./data/corpus_1k_with_variants.jsonl"
OUT_FILE = "./data/minhash_lsh_1kpairs_b125r8.csv"


MODE = "word"      # "char" hoặc "word"
Q = 3              # char: 5; word: 3
T = 0.55           # ngưỡng tương đồng
SEED = 42

# ----- Cấu hình k, b, r -----
# Nhập tay: gán số cụ thể. Để None thì dùng hàm.
K = 1000            # số hàm băm; None -> optimal_k(EPS, DELTA)
EPS, DELTA = 0.05, 0.05
B = 125             # số băng;   None -> chọn bằng BR_METHOD
R = 8              # số hàng/băng; None -> chọn bằng BR_METHOD
BR_METHOD = "optimal"   # "optimal" (cực tiểu FP+FN) hoặc "nearest" (θ gần t nhất)
W_FP, W_FN = 1.0, 1.0   # trọng số cho "optimal"

shingle_fn = char_shingles if MODE == "char" else word_shingles

# 0. Xác định k, b, r
if K is None:
    K = optimal_k(EPS, DELTA)
if B is None or R is None:
    if BR_METHOD == "optimal":
        B, R = optimal_br(K, T, W_FP, W_FN)
    else:
        B, R = nearest_threshold_br(K, T)
if B * R > K:
    raise ValueError(f"b*r = {B * R} vượt quá k = {K}")
theta = (1 / B) ** (1 / R)
print(f"k={K}, b={B}, r={R}, theta={theta:.4f}")

# Đọc corpus
texts = []
with open(IN_FILE, encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
n = len(texts)

# Shingling
t0 = time.time()
shingles = [hash_shingles(shingle_fn(t, Q)) for t in texts]
print(f"Shingle xong: {time.time() - t0:.1f}s")

# MinHash
t1 = time.time()
mh = MinHasher(k=K, seed=SEED)
sigs = mh.signatures(shingles)
print(f"MinHash xong: {time.time() - t1:.1f}s")

# LSH
t2 = time.time()
cands = lsh_candidates(sigs, B, R)
print(f"LSH xong: {len(cands)} cặp ứng viên, {time.time() - t2:.1f}s")

# Kiểm tra Jaccard
t3 = time.time()
pairs = []
for i, j in cands:
    jac = exact_jaccard(shingles[i], shingles[j])
    pairs.append((i, j, jac))
    #if jac >= T: pairs.append((i, j, jac))
total = time.time() - t0
print(f"Kiểm tra xong: {len(pairs)} cặp, {time.time() - t3:.1f}s")

# Ghi file
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    f.write(f"# method=minhash_lsh, mode={MODE}, q={Q}, t={T}, k={K}, b={B}, r={R}, "
            f"theta={theta:.4f}, seed={SEED}, n_docs={n}, "
            f"num_candidates={len(cands)}, "
            f"time={total:.1f}s\n")
    writer = csv.writer(f)
    writer.writerow(["pair", "exact_jaccard"])
    for i, j, jac in pairs:
        writer.writerow([f"{i}-{j}", f"{jac:.6f}"])

print(f"\n{OUT_FILE}: {len(pairs)} cặp có J >= {T}")
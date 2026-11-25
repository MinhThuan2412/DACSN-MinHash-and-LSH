import csv
import json
import time
import numpy as np

from minHash_LSH import char_shingles, word_shingles, hash_shingles, exact_jaccard

IN_FILE = "./data/corpus_1k_with_variants.jsonl"
OUT_FILE = "./data/brute_force_1kpairs.csv"

MODE = "word"      # "char" hoặc "word"
Q = 3              # char: 5; word: 3
T = 0.55           # ngưỡng tương đồng


shingle_fn = char_shingles if MODE == "char" else word_shingles

# Đọc corpus
texts = []
with open(IN_FILE, encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])
n = len(texts)

# Shingling
t0 = time.time()
shingles = [hash_shingles(shingle_fn(t, Q)) for t in texts]
print(f"Shingle xong: {time.time() - t0:.1f}s\n")

# Vét cạn
t1 = time.time()
pairs = []
for i in range(n-1):
    for j in range(i+1,n):
        jac = exact_jaccard(shingles[i],shingles[j])
        if jac >= T: pairs.append([i,j,jac])
total = time.time() - t0
print(f"Vét cạn xong: {n} văn bản, {time.time() - t1:.1f}s\n")

# Ghi file
with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    f.write(f"# method=brute_force, mode={MODE}, q={Q}, t={T}, "
            f"n_docs={n}, condition=J>=t, num_pairs={len(pairs)}, time={total:.1f}s\n")
    writer = csv.writer(f)
    writer.writerow(["pair", "exact_jaccard"])
    for i, j, jac in pairs:
        writer.writerow([f"{i}-{j}", f"{jac:.6f}"])

print(f"\n{OUT_FILE}: {len(pairs)} cặp có J >= {T}")
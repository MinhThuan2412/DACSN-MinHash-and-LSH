import csv
import json

from minHash import (MinHasher, char_shingles, word_shingles, hash_shingles,
                     estimate_jaccard, exact_jaccard)

IN_FILE = "./data/corpus_10k_with_variants.jsonl"
OUT_FILE = "./data/jaccard_check.csv"

MODE = "word"      # "char" hoặc "word"
Q = 3              # char: 5; word: 3
K = 128            # số hàm băm MinHash
SEED = 42
N_PAIRS = 1000     

shingle_fn = char_shingles if MODE == "char" else word_shingles

#Đọc corpus
originals = {}
variants = {}
with open(IN_FILE, encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["type"] == "original":
            originals[d["id"]] = d["text"]
        else:
            variants[d["id"]] = d["text"]

pair_ids = [doc_id for doc_id in originals if doc_id in variants][:N_PAIRS]

# Jaccard
mh = MinHasher(k=K, seed=SEED)

with open(OUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    f.write(f"# k={K}, mode={MODE}, q={Q}, seed={SEED}\n")
    writer.writerow(["index", "estimate_jaccard", "exact_jaccard"])

    for idx, doc_id in enumerate(pair_ids):
        s_o = hash_shingles(shingle_fn(originals[doc_id], Q))
        s_v = hash_shingles(shingle_fn(variants[doc_id], Q))

        sig_o = mh.signature(s_o)
        sig_v = mh.signature(s_v)

        est = estimate_jaccard(sig_o, sig_v)
        exact = exact_jaccard(s_o, s_v)
        writer.writerow([idx, f"{est:.6f}", f"{exact:.6f}"])

print(OUT_FILE)
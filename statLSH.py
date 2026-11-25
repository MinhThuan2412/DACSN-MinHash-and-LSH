import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

BRUTE_FILE = "./data/brute_force_1kpairs.csv"
LSH_FILE = [
    "./data/minhash_lsh_1kpairs_b125r8.csv",
    "./data/minhash_lsh_1kpairs_b100r10.csv",
    "./data/minhash_lsh_1kpairs_b200r5.csv",
]

with open(BRUTE_FILE, encoding="utf-8") as f:
    T = float(re.search(r"t=([+-]?\d+(?:\.\d+)?)", f.readline()).group(1))
bf = pd.read_csv(BRUTE_FILE, comment="#")


def evaluate(truth, df, t):
    cands = set(df["pair"])
    found = set(df.loc[df["exact_jaccard"] >= t, "pair"])   # sau kiểm chứng
    tp = len(found & truth)
    return {
        "num_candidates": len(cands),
        "num_found": len(found),
        "tp": tp,
        "fp": len(cands - found),
        "fn": len(truth - found),
        "candidate_recall": tp / len(truth) if truth else 1.0,
        "candidate_precision": tp / len(cands) if cands else 1.0,
    }


truth = set(bf["pair"])
for path in LSH_FILE:    
    with open(path, encoding="utf-8") as f:
        note = f.readline().strip()
    df = pd.read_csv(path, comment="#")
    res = evaluate(truth, df, T)

    print(note)
    print(f"Vét cạn: {len(truth)} cặp có J >= {T}")
    print(f"Ứng viên LSH       : {res['num_candidates']}")
    print(f"Cặp sau kiểm chứng : {res['num_found']} (TP={res['tp']}, FP={res['fp']}, FN={res['fn']})")
    print(f"Candidate recall   : {res['candidate_recall']:.4f}  (độ phủ của LSH, trước kiểm chứng)")
    print(f"Candidate precision: {res['candidate_precision']:.4f}  (tỉ lệ ứng viên là cặp thật)\n")
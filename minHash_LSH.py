import numpy as np
import pandas as pd
import math
import json

P = (1 << 31) - 1  

# Shingling
def char_shingles(text, q = 5):
    out = []
    for i in range(len(text) - q + 1):
        out.append(text[i:i+q])
    return set(out)

def word_shingles(text, q = 3):
    out = []
    words = text.split()

    for i in range(len(words) - q + 1):
        out.append(" ".join(words[i:i+q]))
    return set(out)


# Băm shingle: FNV-1a hash
def hash_shingle(s):
    h = 14695981039346656037                  
    for byte in s.encode("utf-8"):
        h ^= byte                             
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h % P                              

def hash_shingles(shingles):
    out = []
    for s in shingles:
        out.append(hash_shingle(s))
    arr = np.array(out, dtype=np.uint32)
    return np.unique(arr)

# MinHash
class MinHasher:
    def __init__(self, k = 128, seed = 42):
        rng = np.random.default_rng(seed)
        self.k = k
        self.a = rng.integers(1, P, size=k, dtype=np.uint64)
        self.b = rng.integers(0, P, size=k, dtype=np.uint64)

    def signature(self, hashed, chunk = 20000):
        sig = np.full(self.k, P, dtype=np.uint64)
        if hashed.size == 0:
            return sig.astype(np.uint32)
        a, b = self.a[:, None], self.b[:, None] #chuyển a,b thành shape(k,1)
        for i in range(0, hashed.size, chunk):
            x = hashed[i:i + chunk].astype(np.uint64)[None, :]    #tách shingle thành chunk có shape(1,chunk)
            sig = np.minimum(sig, ((a * x + b) % P).min(axis=1))
        return sig.astype(np.uint32)

    def signatures(self, docs_hashed):
        return np.stack([self.signature(h) for h in docs_hashed]) #shape (số văn bản, k)

# Jaccard
def estimate_jaccard(sig1, sig2):
    return float(np.mean(sig1 == sig2))


def exact_jaccard(a, b):
    union = len(np.union1d(a, b))
    if union == 0:               
        return 1.0
    return len(np.intersect1d(a, b)) / union

# LSH
def lsh_candidates(sigs, b, r):
    cands = set()
    for band in range(b):
        start = band * r
        end = (band + 1) * r
        buckets = {}
        for doc in range(len(sigs)):
            key = tuple(sigs[doc][start:end])      
            if key not in buckets:
                buckets[key] = []
            buckets[key].append(doc)

        for members in buckets.values():
            m = len(members)
            for x in range(m):
                for y in range(x + 1, m):
                    cands.add((members[x], members[y]))
    return cands

# Tìm k
def optimal_k(eps, delta):
    #Hoeffding: |Ĵ - J| <= eps với xác suất >= 1 - delta.
    #k >= ln(2/delta) / (2 eps^2)
    return math.ceil(math.log(2 / delta) / (2 * (eps ** 2)))

# Tìm cặp b, r
def p_candidate(s, b, r):
    return 1 - (1 - s ** r) ** b

def false_positive_area(b, r, t, n=2000):
    s = np.linspace(0.0, t, n)
    y = p_candidate(s, b, r)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(s)) / 2)

def false_negative_area(b, r, t, n=2000):
    s = np.linspace(t, 1.0, n)
    y = 1 - p_candidate(s, b, r)
    return float(np.sum((y[1:] + y[:-1]) * np.diff(s)) / 2)

def optimal_br(k, t, w_fp=1.0, w_fn=1.0, top=0):
    # w_fp, w_fn    : Trọng số của false positive và false negative
    # top > 0       : Trả về danh sách 'top' cấu hình
    results = []
    for b in range(1, k + 1):
        for r in range(1, k // b + 1):
            fp = false_positive_area(b, r, t)
            fn = false_negative_area(b, r, t)
            err = w_fp * fp + w_fn * fn
            results.append((err, b, r, fp, fn))
    results.sort()
    err, b, r, fp, fn = results[0]
    best = (b, r)
    if top:
        return best, [(bb, rr, round(ff, 4), round(nn, 4), round(e, 4))
                      for e, bb, rr, ff, nn in results[:top]]
    return best

def nearest_threshold_br(k, t, top=0):
    # Chọn (b, r) sao cho ngưỡng θ = (1/b)^(1/r) gần t nhất.
    # top > 0      : Trả về danh sách `top` cấu hình
    results = []
    for b in range(1, k + 1):
        for r in range(1, k // b + 1):
            theta = (1 / b) ** (1 / r)
            # hòa độ lệch thì ưu tiên b*r lớn hơn (dùng nhiều hàm băm hơn)
            results.append((abs(theta - t), -b * r, b, r, theta))
    results.sort()
    _, _, b, r, theta = results[0]
    best = (b, r)
    if top:
        return best, [(bb, rr, round(th, 4), round(d, 4))
                      for d, _, bb, rr, th in results[:top]]
    return best



###
# IN_FILE = "./data/corpus_1k_with_variants.jsonl"
# with open(IN_FILE, encoding="utf-8") as f:
#     docs = [json.loads(line) for line in f]
# T=0.55
# text =[]
# for t in docs: text.append(t["text"]) 

# shingles = [hash_shingles(word_shingles(t)) for t in text]
# mh = MinHasher(k=100, seed=42)
# sigs = mh.signatures(shingles)            # shape (2n, K)


# truth = set()
# exact = {}
# for i in range(1000):
#     j = exact_jaccard(shingles[i], shingles[1000 + i])
#     exact[(i, 1000 + i)] = j
#     if j >= T:
#         truth.add((i, 1000 + i))
# print(f"Số văn bản: {len(docs)} | cặp thật (J >= {T}): {len(truth)}")
# #print(truth)
# cand=lsh_candidates(sigs,20,5)
# print(len(cand))
# idx=0
# for i in cand:
#     if i not in truth: 
#         print(f"index {idx}: {i} | {exact_jaccard(shingles[i[0]], shingles[i[1]])}")
#         idx+=1

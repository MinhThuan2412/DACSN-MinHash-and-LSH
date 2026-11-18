import numpy as np
import pandas as pd
import json

P = (1 << 31) - 1  #dtype = np.uint32

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
        self.a = rng.integers(1, P, size=k, dtype=np.uint32)
        self.b = rng.integers(0, P, size=k, dtype=np.uint32)

    def signature(self, hashed, chunk = 20000):
        sig = np.full(self.k, P, dtype=np.uint32)
        if hashed.size == 0:
            return sig
        a, b = self.a[:, None], self.b[:, None] #chuyển a,b thành shape(k,1)
        for i in range(0, hashed.size, chunk):
            x = hashed[i:i + chunk][None, :]    #tách shingle thành chunk có shape(1,chunk)
            sig = np.minimum(sig, ((a * x + b) % P).min(axis=1))
        return sig

    def signatures(self, docs_hashed):
        return np.stack([self.signature(h) for h in docs_hashed])

# Jaccard
def estimate_jaccard(sig1, sig2):
    return float(np.mean(sig1 == sig2))


def exact_jaccard(a, b):
    return len(np.intersect1d(a, b)) / len(np.union1d(a,b))


# IN_FILE = "./data/corpus_1k_with_variants.jsonl"
# with open(IN_FILE, encoding="utf-8") as f:
#     docs = [json.loads(line) for line in f]
# s1 = hash_shingles(word_shingles(docs[1]["text"]))
# s2 = hash_shingles(word_shingles(docs[1001]["text"]))
# mh = MinHasher(128,47)
# sig1 = mh.signature(s1)
# sig2 = mh.signature(s2)
# print(estimate_jaccard(sig1, sig2))
# print(exact_jaccard(s1, s2))

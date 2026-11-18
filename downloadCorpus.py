import json
import os
import random
import re
import unicodedata
import pandas as pd
from datasets import load_dataset
#Link corpus: https://huggingface.co/datasets/bkai-foundation-models/BKAINewsCorpus

OUT_DIR = "./data" 
os.makedirs(OUT_DIR, exist_ok=True)

OUT_FILE = os.path.join(OUT_DIR, "corpus_10k.jsonl")
OUT_GT = os.path.join(OUT_DIR, "corpus_10k_with_variants.jsonl")

N = 10000            # Số bài cần lấy
MIN_WORDS = 150     # Số từ tối thiểu
SEED = 47
BUFFER = 20000

N_VARIANTS = 10000   # Số biến thể (n <= số bản tin)
P_LIST = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5]  # Mức tỉ lệ biến đổi


def normalize(text):
    text = unicodedata.normalize("NFC", text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def mutate(words, p, vocab, rng):
    """
    words: mảng các từ (từ văn bản tách thành)
    p: tỉ lệ từ đó thay đổi
    vocab: tập hợp các từ để thay
    """
    out = []
    for w in words:
        if rng.random() < p:
            op = rng.choice(("sub", "del", "ins"))
            if op == "sub":
                out.append(rng.choice(vocab))
            elif op == "ins":
                out.extend([w, rng.choice(vocab)])
            # "del": bỏ qua w
        else:
            out.append(w)
    return out


#Tải file
def download_corpus():
    ds = load_dataset(
        "bkai-foundation-models/BKAINewsCorpus",
        split="train",
        streaming=True
    )
    ds = ds.shuffle(seed=SEED, buffer_size=BUFFER)

    done = 0
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, encoding="utf-8") as f:
            for line in f: done += 1
        print("Đã tồn tại ", done," bài")

    count = done
    if count < N:
        with open(OUT_FILE, "a", encoding="utf-8") as f:
            for row in ds:
                text = row["text"]
                if len(text.split()) < MIN_WORDS:
                    continue
                f.write(
                    json.dumps({"id": count, "text": text}, ensure_ascii=False) + "\n"
                )
                count += 1
                if count % 200 == 0:
                    f.flush()
                    print(f"{count}/{N}")
                if count >= N:
                    break

    print(f"{OUT_FILE}\n")



def create_ground_truth():
    global N_VARIANTS
    print("Tạo Ground Truth")

    # Đọc corpus
    docs = []
    with open(OUT_FILE, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    if N_VARIANTS > len(docs):
        N_VARIANTS = len(docs)
        print("Điều chỉnh N_VARIANTS =", len(docs))

    # Tạo vocab: 2 từ ngẫu nhiên mỗi bài tin (có thể trùng)
    rng_vocab = random.Random(SEED)
    vocab = []
    for d in docs:
        vocab.extend(rng_vocab.sample(normalize(d["text"]).split(), k=2))

    # Ghi file
    with open(OUT_GT, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(
                json.dumps(
                    {"id": d["id"], "type": "original", "text": normalize(d["text"])},
                    ensure_ascii=False,
                )
                + "\n"
            )

        for k, d in enumerate(docs[:N_VARIANTS]):
            p = P_LIST[k % len(P_LIST)]
            var_seed = SEED * 100000 + k
            words = normalize(d["text"]).split()
            var_text = " ".join(mutate(words, p, vocab, random.Random(var_seed)))
            f.write(
                json.dumps(
                    {"id": d["id"], "type": f"variant_{p}", "text": var_text},
                    ensure_ascii=False,
                )
                + "\n"
            )


    print(f"{OUT_GT} | {len(docs)} gốc + {N_VARIANTS} biến thể\n")



if __name__ == "__main__":
    download_corpus()
    create_ground_truth()
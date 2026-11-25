## Cài đặt

Yêu cầu: Python 3.9+, kết nối Internet (lần đầu tải corpus từ Hugging Face).

```bash
git clone https://github.com/MinhThuan2412/DACSN-MinHash-and-LSH.git
cd DACSN-MinHash-and-LSH
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install numpy pandas matplotlib datasets
```

## Các file chính

| File | Chức năng |
| --- | --- |
| `downloadCorpus.py` | Tải 10.000 bài từ BKAINewsCorpus, chuẩn hóa, tạo bản biến thể làm ground truth |
| `minHash_LSH.py` | Thư viện dùng chung: shingling (ký tự/từ), băm shingle (FNV-1a), `MinHasher`, Jaccard ước lượng và chính xác, `lsh_candidates` (banding), chọn `k` (Hoeffding), chọn `(b, r)` |
| `jaccardCheck.py` | So sánh Ĵ (MinHash) với J thật trên các cặp gốc – biến thể, ghi ra CSV |
| `stat.py` | Thống kê và vẽ biểu đồ sai số của Ĵ, so với độ lệch chuẩn lý thuyết |
| `bruteForce.py` | Vét cạn: tính J chính xác cho mọi cặp, giữ các cặp J ≥ t (đáp án chuẩn cho LSH) |
| `lshCheck.py` | Quy trình MinHash + LSH: tạo chữ ký, sinh cặp ứng viên, tính lại J thật cho ứng viên |
| `statLSH.py` | Đánh giá LSH so với vét cạn: số ứng viên, TP/FP/FN, candidate recall và precision |

## Cách chạy

Chạy các lệnh từ thư mục chứa các file `.py` (các script import `minHash_LSH`).

**Bước 1. Tải corpus và tạo ground truth**

```bash
python downloadCorpus.py
```

Kết quả trong `./data/`:
- `corpus_10k.jsonl`: 10.000 bài gốc (≥ 150 từ), mỗi dòng `{"id", "text"}`.
- `corpus_10k_with_variants.jsonl`: 10.000 bài gốc đã chuẩn hóa (`type = original`) cùng 10.000 biến thể (`type = variant_p`). Mỗi biến thể thay/xóa/chèn từ với tỉ lệ `p` lấy lần lượt trong `[0.02, 0.05, 0.1, 0.2, 0.3, 0.5]`.

Trong `./data` đã có sẵn mẫu:
- `corpus_1k_with_variants.jsonl`: 1.000 bài gốc (dòng 0–999) và 1.000 biến thể tương ứng (dòng 1000–1999, biến thể của bài `i` nằm ở dòng `1000 + i`).

**Bước 2. Tính Ĵ và J thật cho các cặp** (kiểm tra sai số MinHash)

Trong `jaccardCheck.py`, sửa đường dẫn đầu vào cho khớp file vừa tạo:

```python
IN_FILE = "./data/corpus_10k_with_variants.jsonl"
```

Các tham số có thể chỉnh ở đầu file:

| Tham số | Ý nghĩa | Gợi ý |
| --- | --- | --- |
| `MODE` | `"char"` hoặc `"word"` | |
| `Q` | độ dài shingle | char: 5, word: 3 |
| `K` | số hàm băm MinHash | 128 |
| `SEED` | seed cho hàm băm | 42 |
| `N_PAIRS` | số cặp so sánh | 1000 |

```bash
python jaccardCheck.py
```

Kết quả: `./data/jaccard_check.csv` (dòng đầu là chú thích `# k=..., mode=..., q=..., seed=...`, tiếp theo là các cột `index, estimate_jaccard, exact_jaccard`).

**Bước 3. Thống kê sai số**

```bash
python stat.py
```

In ra số cặp, sai số trung bình (bias), độ lệch chuẩn thực nghiệm so với lý thuyết `sqrt(J(1-J)/k)`, tỉ lệ cặp nằm trong 1σ và 2σ, và hiện 3 biểu đồ: histogram sai số, sai số chuẩn hóa so với N(0,1), Ĵ theo J kèm dải ±2σ. Muốn lưu ảnh, bỏ comment dòng `plt.savefig(...)` ở cuối file.

**Bước 4. Vét cạn (đáp án chuẩn)**

```bash
python bruteForce.py
```

Đọc `./data/corpus_1k_with_variants.jsonl` (2.000 văn bản), tính J chính xác cho mọi cặp (≈ 2 triệu cặp, độ phức tạp O(n²·L)) và giữ các cặp có J ≥ `T`.

| Tham số | Ý nghĩa | Giá trị hiện dùng |
| --- | --- | --- |
| `MODE`, `Q` | mức và độ dài shingle | `"word"`, 3 |
| `T` | ngưỡng tương đồng t | 0.55 |

Kết quả: `./data/brute_force_1kpairs.csv`. Dòng đầu là chú thích (`method, mode, q, t, n_docs, num_pairs, time`), tiếp theo là các cột `pair, exact_jaccard` với `pair` có dạng `(i,j)`.

**Bước 5. MinHash + LSH**

```bash
python lshCheck.py
```

Quy trình: shingling → chữ ký MinHash → banding sinh cặp ứng viên → tính lại J thật cho từng cặp ứng viên. In ra thời gian từng giai đoạn và số cặp ứng viên.

| Tham số | Ý nghĩa | Ghi chú |
| --- | --- | --- |
| `MODE`, `Q`, `T`, `SEED` | như trên | nên trùng với `bruteForce.py` |
| `K` | số hàm băm | số cụ thể, hoặc `None` để dùng `optimal_k(EPS, DELTA)` (Hoeffding) |
| `EPS`, `DELTA` | sai số tuyệt đối và xác suất lỗi cho `optimal_k` | 0.05, 0.05 |
| `B`, `R` | số băng và số hàng mỗi băng (`B·R ≤ K`) | số cụ thể, hoặc `None` để chọn tự động |
| `BR_METHOD` | cách chọn `(B, R)` khi để `None` | `"optimal"`: cực tiểu diện tích FP + FN; `"nearest"`: ngưỡng θ = (1/b)^(1/r) gần t nhất |
| `W_FP`, `W_FN` | trọng số FP và FN cho `"optimal"` | 1.0, 1.0 |
| `OUT_FILE` | tên file kết quả | đặt theo cấu hình, ví dụ `minhash_lsh_1kpairs_b125r8.csv` |

Kết quả: file CSV có dòng chú thích (`k, b, r, theta, num_candidates, time, ...`) và các cột `pair, exact_jaccard`. File này lưu **toàn bộ cặp ứng viên** cùng J thật của chúng; việc lọc J ≥ t làm ở bước đánh giá.

Để so sánh nhiều cấu hình, sửa `B`, `R` (và `OUT_FILE`) rồi chạy lại. Các cấu hình đã chạy với `K = 1000`: (b=125, r=8), (b=100, r=10), (b=200, r=5).

**Bước 6. Đánh giá LSH so với vét cạn**

Trong `statLSH.py`, khai báo các file cần so sánh:

```python
BRUTE_FILE = "./data/brute_force_1kpairs.csv"
LSH_FILE = [
    "./data/minhash_lsh_1kpairs_b125r8.csv",
    "./data/minhash_lsh_1kpairs_b100r10.csv",
    "./data/minhash_lsh_1kpairs_b200r5.csv",
]
```

```bash
python statLSH.py
```

Với mỗi cấu hình, in ra: số cặp ứng viên, số cặp sau kiểm chứng (TP, FP, FN so với vét cạn), `candidate recall` (độ phủ của LSH trước kiểm chứng) và `candidate precision` (tỉ lệ ứng viên là cặp thật). Ngưỡng `t` được đọc từ dòng chú thích của file vét cạn.

## Lưu ý

- `stat.py` đọc `k` từ dòng chú thích của CSV nên không cần sửa tay khi đổi `K`.
- Để so sánh ký tự và từ, chạy `jaccardCheck.py` lần lượt với `MODE` khác nhau (nhớ đổi `Q` tương ứng) rồi chạy lại `stat.py`.
- `minHash.py` đã được đổi tên và mở rộng thành `minHash_LSH.py`.
- `bruteForce.py` và `lshCheck.py` phải dùng cùng `MODE`, `Q`, `T` và cùng file corpus thì kết quả mới so sánh được.
- FN trong `statLSH.py` chỉ do LSH bỏ lỡ (không thể khôi phục ở bước kiểm chứng); FP của ứng viên là các cặp J < t, được loại ở bước kiểm chứng.
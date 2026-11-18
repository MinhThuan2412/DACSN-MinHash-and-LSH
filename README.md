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
| `minHash.py` | Shingling (ký tự/từ), băm shingle (FNV-1a), `MinHasher`, Jaccard ước lượng và chính xác |
| `jaccardCheck.py` | So sánh Ĵ (MinHash) với J thật trên các cặp gốc – biến thể, ghi ra CSV |
| `stat.py` | Thống kê và vẽ biểu đồ sai số, so với độ lệch chuẩn lý thuyết |

## Cách chạy

Chạy các lệnh từ thư mục chứa các file `.py` (vì `jaccardCheck.py` import `minHash`).

**Bước 1. Tải corpus và tạo ground truth**

```bash
python downloadCorpus.py
```

Kết quả trong `./data/`:
- `corpus_10k.jsonl`: 10.000 bài gốc (≥ 150 từ), mỗi dòng `{"id", "text"}`.
- `corpus_10k_with_variants.jsonl`: 10.000 bài gốc đã chuẩn hóa (`type = original`) cùng 10.000 biến thể (`type = variant_p`). Mỗi biến thể thay/xóa/chèn từ với tỉ lệ `p` lấy lần lượt trong `[0.02, 0.05, 0.1, 0.2, 0.3, 0.5]`.

Trong `./data` đã có sẵn mẫu

**Bước 2. Tính Ĵ và J thật cho các cặp**

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

## Lưu ý

- `stat.py` đọc `k` từ dòng chú thích của CSV nên không cần sửa tay khi đổi `K`.
- Để so sánh ký tự và từ, chạy `jaccardCheck.py` lần lượt với `MODE` khác nhau (nhớ đổi `Q` tương ứng) rồi chạy lại `stat.py`.

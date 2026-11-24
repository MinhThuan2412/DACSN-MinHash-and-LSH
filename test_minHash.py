import numpy as np
import pytest
from minHash_LSH import (P, MinHasher, char_shingles, word_shingles,
                     hash_shingle, hash_shingles,
                     estimate_jaccard, exact_jaccard)

# Shingling
@pytest.mark.parametrize("s, expected", [
    ("abcdabd", {"ab", "bc", "cd", "da", "bd"}),
    ("abc dabd", {"ab", "bc","c ", " d", "da", "bd"}),
    ("Cảm ơn", {"Cả", "ảm", "m ", " ơ", "ơn"})
])
def test_char_shingles_example(s,expected):
    assert char_shingles(s, 2) == expected

@pytest.mark.parametrize("S, expected", [
    ("Hôm nay trời đẹp lắm", {"Hôm nay trời", "nay trời đẹp", "trời đẹp lắm"}),
    ("Trường Đại học Nha Trang", {"Trường Đại học", "Đại học Nha","học Nha Trang"}),
    ("Công nghệ thông tin", {"Công nghệ thông", "nghệ thông tin"})
])
def test_word_shingles_example(S, expected):
    assert word_shingles(S, 3) == expected

def test_shingles_text_shorter_than_q():
    assert char_shingles("abc", 5) == set()
    assert word_shingles("một hai", 3) == set()

# Băm shingle
def test_hash_shingle_deterministic_and_in_range():
    h = hash_shingle("hôm nay trời")
    assert h == hash_shingle("hôm nay trời")
    assert 0 <= h < P

def test_hash_shingles_sorted_unique():
    arr = hash_shingles({"a b c", "b c d", "c d e", "a b c"})
    assert np.array_equal(arr, np.unique(arr))

# Tràn số
def test_signature_matches_bigint_reference():
    rng = np.random.default_rng(0)
    x = np.unique(rng.integers(0, P, size=5000).astype(np.uint32))
    mh = MinHasher(k=64, seed=1)
    sig = mh.signature(x)
    for i in range(mh.k):
        a, b = int(mh.a[i]), int(mh.b[i])
        expected = min((a * int(v) + b) % P for v in x)  # số nguyên Python, không tràn
        assert int(sig[i]) == expected, f"hàm băm {i} sai (nghi tràn số)"

def test_signature_values_in_range():
    x = np.arange(1, 1000, dtype=np.uint32) * 2_000_003 % P
    sig = MinHasher(k=32, seed=3).signature(np.unique(x))
    assert sig.max() < P

# MinHash
def test_signature_chunk_invariance():
    x = np.unique(np.random.default_rng(5).integers(0, P, 1000).astype(np.uint32))
    mh = MinHasher(k=32, seed=2)
    assert np.array_equal(mh.signature(x, chunk=7), mh.signature(x, chunk=100000))

def test_signature_reproducible_with_seed():
    x = np.arange(100, dtype=np.uint32)
    assert np.array_equal(MinHasher(16, 9).signature(x), MinHasher(16, 9).signature(x))

def test_empty_set_signature():
    sig = MinHasher(k=8).signature(np.array([], dtype=np.uint32))
    assert (sig == P).all()

def test_identical_sets_estimate_one():
    x = np.arange(500, dtype=np.uint32)
    mh = MinHasher(k=64)
    assert estimate_jaccard(mh.signature(x), mh.signature(x)) == 1.0

def test_disjoint_sets_estimate_near_zero():
    mh = MinHasher(k=256, seed=7)
    a = np.arange(0, 1000, dtype=np.uint32)
    b = np.arange(10**6, 10**6 + 1000, dtype=np.uint32)
    assert estimate_jaccard(mh.signature(a), mh.signature(b)) < 0.05

@pytest.mark.parametrize("k", [1, 16, 128, 1000])
def test_signature_length_and_dtype(k):
    x = np.arange(1, 300, dtype=np.uint32)
    mh = MinHasher(k=k, seed=5)
    sig = mh.signature(x)
    assert sig.shape == (k,)
    assert sig.dtype == np.uint32
    assert mh.signature(np.array([], dtype=np.uint32)).shape == (k,)

def test_signatures_matrix_shape():
    docs = [np.arange(1, 50, dtype=np.uint32),
            np.arange(30, 90, dtype=np.uint32),
            np.arange(100, 130, dtype=np.uint32)]
    sigs = MinHasher(k=32, seed=1).signatures(docs)
    assert sigs.shape == (3, 32) 

# Jaccard chính xác
@pytest.mark.parametrize("A, B, expected", [
    ([0, 3],    [0, 2, 3],    2 / 3),   
    ([0, 3],    [1, 3, 4],    1 / 4),   
    ([2],       [0, 3],       0.0),     
    ([1, 2, 3], [1, 2, 3],    1.0),     
    ([1, 2],    [1, 2, 3, 4], 2 / 4),
    ([],        [],           1)  
])
def test_exact_jaccard_hand_computed(A, B, expected):
    A = np.array(A, dtype=np.uint32)
    B = np.array(B, dtype=np.uint32)
    assert exact_jaccard(A, B) == pytest.approx(expected)
    assert exact_jaccard(B, A) == pytest.approx(expected) 

# Thống kê
@pytest.mark.parametrize("k", [64, 256, 1024])
def test_estimator_within_4_sigma(k):
    # |A∩B|=500, |A∪B|=1000 -> J = 0.5
    A = np.arange(0, 750, dtype=np.uint32)
    B = np.arange(250, 1000, dtype=np.uint32)
    J = exact_jaccard(A, B)
    assert J == pytest.approx(0.5)  # J thật = 500/1000 = 0.5
    sigma = np.sqrt(J * (1 - J) / k)
    mh = MinHasher(k=k, seed=65)
    est = estimate_jaccard(mh.signature(A), mh.signature(B))
    assert abs(est - J) < 4 * sigma

def test_estimator_unbiased_over_seeds():
    A = np.arange(0, 750, dtype=np.uint32)
    B = np.arange(250, 1000, dtype=np.uint32)
    J = exact_jaccard(A, B)
    ests = [estimate_jaccard(MinHasher(128, s).signature(A),
                             MinHasher(128, s).signature(B)) for s in range(40)]
    sigma_mean = np.sqrt(J * (1 - J) / 128)
    assert abs(np.mean(ests) - J) < 4 * sigma_mean
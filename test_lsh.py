import numpy as np
import math
import pytest

from minHash_LSH import (
    optimal_k, p_candidate, false_positive_area, false_negative_area,
    optimal_br, nearest_threshold_br,lsh_candidates
)

# LSH
def test_identical_signatures_become_candidates():
    sigs = np.array([
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [9, 8, 7, 6, 5, 4],
    ])
    assert lsh_candidates(sigs, b=3, r=2) == {(0, 1)}


@pytest.mark.parametrize(
    "sig_a, sig_b, expected",
    [
        ([1, 2, 10, 20, 3, 4], [5, 6, 10, 20, 7, 8], {(0, 1)}),
        ([1, 2, 10, 20, 3, 4], [5, 6, 11, 21, 7, 8], set()),
        ([1, 2, 10, 20, 3, 4], [1, 9, 11, 21, 7, 8], set()),
    ],
)
def test_match_requires_full_band(sig_a, sig_b, expected):
    sigs = np.array([sig_a, sig_b])
    assert lsh_candidates(sigs, b=3, r=2) == expected


def test_pairs_are_ordered_and_not_duplicated():
    sigs = np.array([
        [1, 2, 3, 4],
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ])
    cands = lsh_candidates(sigs, b=2, r=2)
    assert cands == {(0, 1), (0, 2), (1, 2)}
    assert all(i < j for i, j in cands)

# Tìm k
def test_optimal_k_matches_report_example():
    #eps = 0,05; delta = 0,05 -> k = 738
    assert optimal_k(0.05, 0.05) == 738


def test_optimal_k_satisfies_inequality_and_is_minimal():
    eps, delta = 0.1, 0.01
    k = optimal_k(eps, delta)
    assert 2 * math.exp(-2 * k * eps ** 2) <= delta
    assert 2 * math.exp(-2 * (k - 1) * eps ** 2) > delta


def test_optimal_k_monotonic():
    assert optimal_k(0.02, 0.05) > optimal_k(0.05, 0.05)   # eps nhỏ hơn -> cần nhiều k hơn
    assert optimal_k(0.05, 0.01) > optimal_k(0.05, 0.05)   # delta nhỏ hơn -> cần nhiều k hơn


# p_candidate
def test_p_candidate_single_hash_is_identity():
    # b = r = 1 thì P(s) = s
    for s in (0.1, 0.5, 0.9):
        assert p_candidate(s, 1, 1) == pytest.approx(s)


@pytest.mark.parametrize("s, expected", [
    (0.2, 0.006), (0.3, 0.048), (0.4, 0.186), (0.0, 0.0),
    (0.5, 0.470), (0.6, 0.802), (0.8, 0.9996), (1.0, 1.0),
])
def test_p_candidate_matches(s, expected):
    assert p_candidate(s, 20, 5) == pytest.approx(expected, abs=1e-3)


# diện tích sai số
def test_error_areas_for_identity_curve():
    t = 0.5
    assert false_positive_area(1, 1, t) == pytest.approx(t ** 2 / 2, abs=1e-4)
    assert false_negative_area(1, 1, t) == pytest.approx((1 - t) ** 2 / 2, abs=1e-4)


def test_false_positive_area_trends():
    t = 0.55
    assert false_positive_area(20, 5, t) > false_positive_area(5, 5, t)    # tăng b -> FP tăng
    assert false_positive_area(20, 10, t) < false_positive_area(20, 5, t)  # tăng r -> FP giảm


def test_false_negative_area_trends():
    t = 0.55
    assert false_negative_area(20, 5, t) < false_negative_area(5, 5, t)    # tăng b -> FN giảm
    assert false_negative_area(20, 10, t) > false_negative_area(20, 5, t)  # tăng r -> FN tăng


# optimal_br
def test_optimal_br_respects_k():
    b, r = optimal_br(k=30, t=0.6)
    assert b >= 1 and r >= 1 and b * r <= 30


def test_optimal_br_is_best_by_total_error():
    k, t = 30, 0.6
    b, r = optimal_br(k, t)
    best = false_positive_area(b, r, t) + false_negative_area(b, r, t)
    for bb in range(1, k + 1):
        for rr in range(1, k // bb + 1):
            err = false_positive_area(bb, rr, t) + false_negative_area(bb, rr, t)
            assert best <= err + 1e-12


def test_optimal_br_top_sorted_by_error():
    best, top = optimal_br(k=20, t=0.5, top=5)
    assert len(top) == 5
    assert (top[0][0], top[0][1]) == best
    errs = [row[4] for row in top]          # (b, r, fp, fn, err)
    assert errs == sorted(errs)


def test_optimal_br_weights_shift_choice():
    # Phạt FN nặng -> ngưỡng đường cong thấp hơn (hoặc bằng) so với phạt FP nặng
    b1, r1 = optimal_br(k=60, t=0.6, w_fp=1.0, w_fn=10.0)
    b2, r2 = optimal_br(k=60, t=0.6, w_fp=10.0, w_fn=1.0)
    theta1 = (1 / b1) ** (1 / r1)
    theta2 = (1 / b2) ** (1 / r2)
    assert theta1 <= theta2


# nearest_threshold_br
def test_nearest_threshold_br_is_closest():
    k, t = 100, 0.55
    b, r = nearest_threshold_br(k, t)
    assert b * r <= k
    d = abs((1 / b) ** (1 / r) - t)
    # không kém hơn cấu hình (20, 5) trong báo cáo (theta ≈ 0,549)
    assert d <= abs((1 / 20) ** (1 / 5) - t) + 1e-12
    assert d < 0.01


def test_nearest_threshold_br_top_sorted_by_distance():
    best, top = nearest_threshold_br(k=100, t=0.55, top=5)
    assert len(top) == 5
    assert (top[0][0], top[0][1]) == best
    dists = [row[3] for row in top]         # (b, r, theta, d)
    assert dists == sorted(dists)


def test_nearest_threshold_br_k_equals_one():
    assert nearest_threshold_br(1, 0.5) == (1, 1)
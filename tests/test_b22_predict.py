import numpy as np
import pytest

from bobsphog.b22_predict import (
    PredictorSuite,
    RouteExample,
    evaluate_selection,
    select_equal_layer_budget,
    simulate_pinned_bundle_lru,
)


def _example(identifier, split, prefill_ids, decode_ids):
    prefill = np.zeros((2, 4), dtype=np.float32)
    decode = np.zeros((2, 4), dtype=np.float32)
    for layer, expert in prefill_ids:
        prefill[layer, expert] = 1
    for layer, expert, count in decode_ids:
        decode[layer, expert] = count
    return RouteExample(identifier, "toy", split, prefill, decode)


def test_equal_layer_budget_selects_exact_quota() -> None:
    scores = np.asarray([[4, 3, 2, 1], [1, 2, 3, 4]], dtype=np.float32)

    selected = select_equal_layer_budget(scores, budget=4)

    assert selected.sum() == 4
    assert selected[0].tolist() == [True, True, False, False]
    assert selected[1].tolist() == [False, False, True, True]

    assert not select_equal_layer_budget(scores, budget=0).any()


def test_conditional_index_associates_prefill_and_decode_pages() -> None:
    training = [
        _example("a", "train", [(0, 0), (1, 0)], [(0, 2, 3), (1, 2, 3)]),
        _example("b", "train", [(0, 1), (1, 1)], [(0, 3, 3), (1, 3, 3)]),
    ]
    query = _example("q", "test", [(0, 0), (1, 0)], [(0, 2, 1), (1, 2, 1)])
    suite = PredictorSuite(training)

    scores = suite.scores("conditional_coactivation", query, alpha=0.0)

    assert scores[0, 2] > scores[0, 3]
    assert scores[1, 2] > scores[1, 3]


def test_selection_metrics_weight_repeated_decode_requests() -> None:
    example = _example(
        "q",
        "test",
        [(0, 0)],
        [(0, 1, 5), (0, 2, 1), (1, 3, 2)],
    )
    selected = np.zeros((2, 4), dtype=bool)
    selected[0, 1] = True
    selected[1, 0] = True

    result = evaluate_selection(example, selected, page_bytes=16)

    assert result["union_recall"] == pytest.approx(1 / 3)
    assert result["request_hit_fraction"] == pytest.approx(5 / 8)
    assert result["late_unique_pages"] == 2
    assert result["late_unique_bytes"] == 32


def test_pinned_bundle_lru_uses_residual_capacity() -> None:
    example = _example("q", "test", [], [(0, 0, 2), (0, 1, 2), (1, 2, 2)])
    example = RouteExample(
        example.id,
        example.domain,
        example.split,
        example.prefill,
        example.decode_counts,
        decode_groups=(
            ((0, 1), (2,)),
            ((0, 1), (2,)),
        ),
    )
    selected = np.zeros((2, 4), dtype=bool)
    selected[0, 0] = True

    result = simulate_pinned_bundle_lru(
        example,
        selected,
        cache_capacity_pages=3,
        page_bytes=16,
    )

    assert result["decode_requests"] == 6
    assert result["pinned_request_hits"] == 2
    assert result["transient_request_hits"] == 2
    assert result["simulated_page_faults"] == 2
    assert result["simulated_fault_bytes"] == 32
    assert result["bundle_prefetch_bytes"] == 16
    assert result["cold_total_transfer_bytes"] == 48
    assert result["effective_cache_hit_fraction"] == pytest.approx(4 / 6)

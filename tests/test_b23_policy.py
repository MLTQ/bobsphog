import numpy as np

from bobsphog.b22_predict import PredictorSuite, RouteExample
from bobsphog.b23_policy import (
    confidence_features,
    fit_ridge,
    route_groups,
    simulate_decode_misses,
)


def _example(identifier, split, prefill_expert, decode_expert):
    prefill = np.zeros((2, 4), dtype=np.float32)
    prefill[:, prefill_expert] = 1
    decode = np.zeros((2, 4), dtype=np.float32)
    decode[:, decode_expert] = 2
    return RouteExample(
        identifier,
        "toy",
        split,
        prefill,
        decode,
        decode_groups=(
            ((decode_expert,), (decode_expert,)),
            ((decode_expert,), (decode_expert,)),
        ),
    )


def test_route_groups_preserve_prefill_and_decode_order() -> None:
    example = _example("a", "test", 0, 1)

    prefill, decode = route_groups(example)

    assert prefill == (((0, 0),), ((1, 0),))
    assert decode[0] == ((0, 1),)
    assert decode[-1] == ((1, 1),)


def test_confidence_features_do_not_require_query_decode_counts() -> None:
    training = [
        _example("a", "train", 0, 1),
        _example("b", "train", 2, 3),
    ]
    query = _example("q", "test", 0, 1)
    suite = PredictorSuite(training)

    first, selected, diagnostics = confidence_features(
        suite, query, 2, cache_pages=6, neighbors=1
    )
    altered = RouteExample(
        query.id,
        query.domain,
        query.split,
        query.prefill,
        np.zeros_like(query.decode_counts),
        decode_groups=query.decode_groups,
    )
    second, _, _ = confidence_features(
        suite, altered, 2, cache_pages=6, neighbors=1
    )

    assert np.array_equal(first, second)
    assert selected.sum() == 2
    assert diagnostics["nearest_similarity"] == 1.0


def test_ridge_calibrator_recovers_linear_targets() -> None:
    features = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    targets = 4.0 + 2.0 * features[:, 0]

    calibrator = fit_ridge(features, targets, alpha=0.0)

    assert np.allclose(calibrator.predict(features), targets)


def test_pinned_simulation_can_reduce_decode_misses() -> None:
    prefill = np.zeros((1, 4), dtype=np.float32)
    decode = np.asarray([[0, 2, 1, 1]], dtype=np.float32)
    example = RouteExample(
        "a",
        "toy",
        "test",
        prefill,
        decode,
        decode_groups=(((1,),), ((2,),), ((3,),), ((1,),)),
    )
    empty = np.zeros_like(example.prefill, dtype=bool)
    selected = empty.copy()
    selected[0, 1] = True

    assert simulate_decode_misses(example, selected, cache_pages=2) < simulate_decode_misses(
        example, empty, cache_pages=2
    )

import pytest

from bobsphog.cache_simulation import (
    simulate_grouped_belady,
    simulate_grouped_lru,
)


def test_grouped_belady_beats_lru_when_future_use_is_known() -> None:
    a = (0, 1)
    b = (0, 2)
    c = (0, 3)
    groups = [(a, b), (c,), (a,)]

    lru = simulate_grouped_lru(groups, capacity_pages=2)
    belady = simulate_grouped_belady(groups, capacity_pages=2)

    assert lru.requests == belady.requests == 4
    assert lru.misses == 4
    assert belady.misses == 3
    assert belady.hit_rate == pytest.approx(0.25)
    assert belady.describe(page_bytes=16)["bytes_transferred"] == 48


def test_grouped_simulations_reject_an_impossible_atomic_working_set() -> None:
    groups = [((0, 1), (0, 2), (0, 3))]

    with pytest.raises(ValueError, match="cannot fit"):
        simulate_grouped_lru(groups, capacity_pages=2)
    with pytest.raises(ValueError, match="cannot fit"):
        simulate_grouped_belady(groups, capacity_pages=2)


def test_duplicate_keys_within_a_group_count_once() -> None:
    key = (0, 1)

    result = simulate_grouped_lru([(key, key), (key,)], capacity_pages=1)

    assert result.requests == 2
    assert result.misses == 1
    assert result.hits == 1


def test_lru_replays_post_schedule_touch_order() -> None:
    a = (0, 1)
    b = (0, 2)
    c = (0, 3)
    d = (0, 4)

    result = simulate_grouped_lru([(a, b), (a, c), (d,), (a,)], capacity_pages=3)

    assert result.misses == 4
    assert result.hits == 2

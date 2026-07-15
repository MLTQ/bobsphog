from bobsphog.paging import PagePlan


def test_uniform_prefix_caps_each_layer() -> None:
    counts = {"a": 1, "b": 3}
    plan = PagePlan.uniform_prefix(counts, pages_per_layer=2)

    assert plan.selected("a", 1) == (0,)
    assert plan.selected("b", 3) == (0, 1)
    assert plan.selected("unknown", 4) == ()


def test_random_dropout_is_reproducible_and_structured() -> None:
    counts = {"a": 8, "b": 8}
    first = PagePlan.random_dropout(counts, 0.5, seed=11)
    second = PagePlan.random_dropout(counts, 0.5, seed=11)

    assert first == second
    assert first.selected("a", 8) != tuple(range(8))


def test_full_and_base_defaults() -> None:
    assert PagePlan.full().selected("layer", 3) == (0, 1, 2)
    assert PagePlan.base_only().selected("layer", 3) == ()

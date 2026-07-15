import pytest

from bobsphog.oracle_cache import FutureUseOracle, OracleTraceMismatch


def test_future_use_oracle_selects_the_page_used_furthest_ahead() -> None:
    a = (0, 1)
    b = (0, 2)
    c = (0, 3)
    oracle = FutureUseOracle([(a, b), (c,), (a,)])

    oracle.consume((a, b))
    oracle.consume((c,))

    assert oracle.choose_victim((a, b), protected=set()) == b
    assert oracle.next_use(a) == 2.0


def test_future_use_oracle_rejects_live_route_divergence() -> None:
    oracle = FutureUseOracle([((0, 1),), ((1, 2),)])

    with pytest.raises(OracleTraceMismatch, match="group 0"):
        oracle.consume(((0, 9),))


def test_future_use_oracle_reports_completion() -> None:
    oracle = FutureUseOracle([((0, 1),)])

    assert not oracle.complete
    oracle.consume(((0, 1),))
    assert oracle.complete

    with pytest.raises(OracleTraceMismatch, match="beyond"):
        oracle.consume(((0, 1),))

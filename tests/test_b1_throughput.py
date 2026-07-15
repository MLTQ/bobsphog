import pytest

from bobsphog.b1_throughput import summarize_decode_latencies


def test_decode_latency_summary_reports_rate_and_percentiles() -> None:
    result = summarize_decode_latencies([0.1, 0.2, 0.3, 0.4])

    assert result["tokens"] == 4
    assert result["seconds"] == pytest.approx(1.0)
    assert result["tokens_per_second"] == pytest.approx(4.0)
    assert result["median_seconds_per_token"] == pytest.approx(0.25)
    assert result["p95_seconds_per_token"] == pytest.approx(0.4)


def test_decode_latency_summary_rejects_invalid_samples() -> None:
    with pytest.raises(ValueError):
        summarize_decode_latencies([])
    with pytest.raises(ValueError):
        summarize_decode_latencies([0.1, 0.0])

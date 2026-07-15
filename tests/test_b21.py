import json

import pytest

from bobsphog.b21 import load_recorded_route, summarize_prefetch_pipeline


def test_load_recorded_route_reconstructs_layer_keys(tmp_path) -> None:
    payload = {
        "config": {"prompt": "test prompt"},
        "runs": [{"selected_token_ids": [11, 12, 13]}],
        "routing_trace": {
            "prefill_experts_by_layer": [[3, 4], [5]],
            "decode_experts_by_token_and_layer": [
                [[4], [6]],
                [[7], [5]],
            ],
        },
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))

    route = load_recorded_route(path)

    assert route.prompt == "test prompt"
    assert route.selected_token_ids == (11, 12, 13)
    assert route.prefill_groups == (((0, 3), (0, 4)), ((1, 5),))
    assert route.decode_traces[0] == (((0, 4),), ((1, 6),))
    assert len(route.groups) == 6
    assert route.decode_union_groups == (
        ((0, 4), (0, 7)),
        ((1, 6), (1, 5)),
    )


def test_load_recorded_route_rejects_misaligned_decode_count(tmp_path) -> None:
    payload = {
        "config": {"prompt": "test"},
        "runs": [{"selected_token_ids": [1, 2, 3]}],
        "routing_trace": {
            "prefill_experts_by_layer": [[1]],
            "decode_experts_by_token_and_layer": [[[1]]],
        },
    }
    path = tmp_path / "trace.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="decode trace count"):
        load_recorded_route(path)


def test_prefetch_pipeline_reports_serial_compute_and_overlap_rates() -> None:
    result = summarize_prefetch_pipeline(
        prefetch_seconds=[2.0, 3.0, 1.0],
        compute_seconds=[1.0, 1.0, 1.0],
    )

    assert result["serial_seconds"] == pytest.approx(9.0)
    assert result["compute_only_seconds"] == pytest.approx(3.0)
    assert result["ideal_one_token_overlap_seconds"] == pytest.approx(7.0)
    assert result["serial_tokens_per_second"] == pytest.approx(1 / 3)
    assert result["compute_only_tokens_per_second"] == pytest.approx(1.0)
    assert result["ideal_one_token_overlap_tokens_per_second"] == pytest.approx(3 / 7)

import json

import pytest

from bobsphog.b22_live import load_live_case, route_to_example


def test_load_live_case_builds_exact_recorded_route(tmp_path) -> None:
    payload = {
        "prompts": [
            {
                "id": "held-out",
                "domain": "toy",
                "split": "test",
                "prompt": "hello",
                "selected_token_ids": [10, 11],
                "prefill_experts_by_layer": [[1, 2], [3]],
                "decode_experts_by_token_and_layer": [[[2], [4]]],
            }
        ]
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload))

    _, example, route = load_live_case(path, "held-out")

    assert example.target.sum() == 2
    assert route.selected_token_ids == (10, 11)
    assert route.prefill_groups == (((0, 1), (0, 2)), ((1, 3),))
    assert route.decode_traces == ((((0, 2),), ((1, 4),)),)

    converted = route_to_example(route, example)
    assert converted.prefill.sum() == 3
    assert converted.decode_counts.sum() == 2
    assert converted.decode_groups == (((2,), (4,)),)


def test_load_live_case_rejects_training_prompt(tmp_path) -> None:
    payload = {
        "prompts": [
            {
                "id": "train",
                "domain": "toy",
                "split": "train",
                "prompt": "hello",
                "selected_token_ids": [10, 11],
                "prefill_experts_by_layer": [[1]],
                "decode_experts_by_token_and_layer": [[[1]]],
            }
        ]
    }
    path = tmp_path / "corpus.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="held-out"):
        load_live_case(path, "train")

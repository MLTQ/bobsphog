import pytest

from bobsphog.b2 import analyze_decode_working_sets


def test_decode_working_set_analysis_reports_growth_and_overlap() -> None:
    first = (((0, 1), (0, 2)), ((1, 3),))
    second = (((0, 2), (0, 4)), ((1, 3), (1, 5)))

    result = analyze_decode_working_sets([first, second], num_layers=2)

    assert result["decode_forwards"] == 2
    assert result["final_cumulative_unique_pages"] == 5
    assert result["tokens"][0]["cumulative_unique_pages"] == 3
    assert result["tokens"][1]["new_pages"] == 2
    transition = result["transitions"][0]
    assert transition["overlap_pages"] == 2
    assert transition["current_working_set_recall_from_previous"] == pytest.approx(0.5)
    assert transition["jaccard"] == pytest.approx(0.4)
    assert result["per_layer"][0]["previous_token_recall"]["mean"] == pytest.approx(
        0.5
    )
    assert result["per_layer"][1]["previous_token_recall"]["mean"] == pytest.approx(
        0.5
    )


def test_decode_working_set_analysis_requires_one_group_per_layer() -> None:
    with pytest.raises(RuntimeError, match="expected 2"):
        analyze_decode_working_sets([(((0, 1),),)], num_layers=2)


def test_decode_working_set_analysis_handles_single_forward() -> None:
    result = analyze_decode_working_sets(
        [(((0, 1),), ((1, 2),))],
        num_layers=2,
    )

    assert result["transitions"] == []
    assert result["previous_token_predictor"]["working_set_recall"]["mean"] is None

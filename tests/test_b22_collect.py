import json

import pytest

from bobsphog.b22_collect import load_prompt_corpus


def test_prompt_corpus_loader_preserves_stratification(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "domain": "science", "split": "train", "prompt": "Why?"},
                {"id": "b", "domain": "code", "split": "test", "prompt": "Fix it."},
            ]
        )
    )

    records = load_prompt_corpus(path)

    assert [record["id"] for record in records] == ["a", "b"]
    assert records[1]["split"] == "test"


def test_prompt_corpus_loader_rejects_duplicate_ids(tmp_path) -> None:
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps(
            [
                {"id": "a", "domain": "x", "split": "train", "prompt": "one"},
                {"id": "a", "domain": "x", "split": "validation", "prompt": "two"},
            ]
        )
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_prompt_corpus(path)


import csv
from pathlib import Path

import pytest

from bobsphog.glm_b22_benchmark import (
    MmluExample,
    format_mmlu_prompt,
    load_mmlu_examples,
    select_prefill_bundle,
    wilson_interval,
)


def _write_subject(root: Path, subject: str, answers: tuple[str, ...]) -> None:
    path = root / "test" / f"{subject}_test.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for index, answer in enumerate(answers):
            writer.writerow([f"Question {index}?", "one", "two", "three", "four", answer])


def test_load_mmlu_examples_is_subject_stratified_and_deterministic(tmp_path: Path) -> None:
    _write_subject(tmp_path, "abstract_algebra", ("A", "B", "C"))
    _write_subject(tmp_path, "world_history", ("D", "C", "B"))

    first = load_mmlu_examples(tmp_path, samples_per_subject=1, seed=72)
    second = load_mmlu_examples(tmp_path, samples_per_subject=1, seed=72)

    assert first == second
    assert [item.subject for item in first] == ["abstract_algebra", "world_history"]
    assert len(first) == 2

    limited = load_mmlu_examples(
        tmp_path, samples_per_subject=1, seed=72, subject_limit=1
    )
    assert len(limited) == 1
    assert limited[0].subject in {"abstract_algebra", "world_history"}


def test_format_and_prefill_bundle_use_direct_choices_and_equal_layer_quota() -> None:
    example = MmluExample(
        subject="computer_security",
        row_index=4,
        question="Which choice?",
        choices=("alpha", "beta", "gamma", "delta"),
        answer="C",
    )
    prompt = format_mmlu_prompt(example)
    assert "Subject: computer security" in prompt
    assert "A. alpha" in prompt
    assert prompt.endswith("Answer:")

    bundle = select_prefill_bundle(
        {3: (0, 7, 2, 7), 4: (9, 1, 8, 0)},
        (3, 4),
        budget=3,
    )
    assert bundle == ((3, 1), (3, 3), (4, 0))


def test_wilson_interval_contains_observed_accuracy() -> None:
    low, high = wilson_interval(40, 57)
    assert low < 40 / 57 < high
    with pytest.raises(ValueError):
        wilson_interval(0, 0)

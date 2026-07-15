import torch

from bobsphog.evaluation import evaluate_random_budget_curve, page_ablation_utilities
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.synthetic import TwoDomainArithmetic


def test_random_budget_curve_and_ablation_cover_page_layout() -> None:
    torch.manual_seed(12)
    task = TwoDomainArithmetic(context_length=9)
    model = ToyTransformer(
        ToyConfig(
            vocab_size=task.VOCAB_SIZE,
            context_length=9,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            base_rank=2,
            page_rank=2,
        )
    )
    curve = evaluate_random_budget_curve(
        model,
        task,
        dropout_rates=(1.0, 0.0),
        batch_size=4,
        batches=1,
        seed=3,
        device=torch.device("cpu"),
    )
    utilities = page_ablation_utilities(
        model,
        task,
        domain="addition",
        batch_size=4,
        seed=4,
        device=torch.device("cpu"),
    )

    assert curve[0]["mean_resident_fraction"] < curve[1]["mean_resident_fraction"]
    assert curve[1]["mean_resident_fraction"] == 1.0
    assert len(utilities) == sum(model.page_counts().values())

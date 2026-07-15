import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.retriever import (
    CounterfactualUtilityEstimator,
    learned_base_query_selection,
    learned_greedy_selection,
    train_utility_estimator,
)
from bobsphog.synthetic import TwoDomainArithmetic
from bobsphog.utility_data import collect_utility_examples


def test_counterfactual_collection_training_and_selection_run() -> None:
    torch.manual_seed(14)
    device = torch.device("cpu")
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
    ).eval()
    catalog = PageCatalog.from_model(model)
    examples = collect_utility_examples(
        model,
        task,
        catalog,
        states=2,
        batch_size=4,
        candidates_per_state=2,
        resident_budgets=(0, 1),
        seed=3,
        device=device,
    )
    estimator = CounterfactualUtilityEstimator(8, len(catalog), hidden_size=8)
    summary = train_utility_estimator(
        estimator,
        examples,
        examples,
        steps=2,
        batch_size=4,
        learning_rate=1e-3,
        seed=4,
        device=device,
    )
    batch = task.sample(
        4,
        generator=torch.Generator().manual_seed(5),
        domain="addition",
    )
    selected = learned_greedy_selection(model, batch, catalog, estimator, budget=2)
    base_selected = learned_base_query_selection(
        model,
        batch,
        catalog,
        estimator,
        budget=2,
    )

    assert len(examples) == 16
    assert torch.isfinite(torch.tensor(summary.validation_rmse))
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert len(base_selected) == 2
    assert len(set(base_selected)) == 2

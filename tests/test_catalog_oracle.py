import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.oracle import greedy_oracle_selection
from bobsphog.synthetic import TwoDomainArithmetic


def make_fixture() -> tuple[ToyTransformer, TwoDomainArithmetic, PageCatalog]:
    torch.manual_seed(13)
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
    return model, task, PageCatalog.from_model(model)


def test_catalog_round_trips_global_ids_to_layer_plan() -> None:
    model, _, catalog = make_fixture()
    selected = catalog.static_prefix(2)
    plan = catalog.plan(selected)

    assert len(catalog) == sum(model.page_counts().values())
    assert catalog.resident_mask(selected).sum().item() == 2
    assert sum(len(plan.selected(layer, count)) for layer, count in model.page_counts().items()) == 2
    assert len(catalog.names(selected)) == 2


def test_greedy_oracle_returns_fixed_budget_and_nonincreasing_loss() -> None:
    model, task, catalog = make_fixture()
    batch = task.sample(
        8,
        generator=torch.Generator().manual_seed(2),
        domain="addition",
    )
    result = greedy_oracle_selection(model, batch, catalog, budget=2)

    assert len(result.selected_ids) == 2
    assert len(set(result.selected_ids)) == 2
    assert all(
        next_loss <= loss + 1e-7
        for loss, next_loss in zip(result.calibration_losses, result.calibration_losses[1:])
    )

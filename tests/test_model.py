import torch

from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.paging import PagePlan, PagingTrace


def make_model() -> ToyTransformer:
    torch.manual_seed(3)
    return ToyTransformer(
        ToyConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            n_heads=4,
            n_layers=2,
            d_ff=32,
            base_rank=2,
            page_rank=2,
        )
    ).eval()


def test_model_runs_full_and_base_only_paths() -> None:
    model = make_model()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    targets = torch.randint(0, model.config.vocab_size, (2, 8))
    base_trace = PagingTrace()

    full = model(input_ids, targets)
    base = model(input_ids, targets, plan=PagePlan.base_only(), trace=base_trace)

    assert full.logits.shape == (2, 8, model.config.vocab_size)
    assert full.loss is not None and torch.isfinite(full.loss)
    assert base.loss is not None and torch.isfinite(base.loss)
    assert not torch.equal(full.logits, base.logits)
    assert base_trace.selected_page_count == 0
    assert len(base_trace.events) == 2 * model.config.n_layers


def test_logical_resident_bytes_grow_to_full_model_size() -> None:
    model = make_model()
    counts = model.page_counts()
    base = PagePlan.base_only()
    full = PagePlan.full()
    prefix = PagePlan.uniform_prefix(counts, pages_per_layer=1)

    assert model.resident_parameter_bytes(base) < model.resident_parameter_bytes(prefix)
    assert model.resident_parameter_bytes(prefix) < model.resident_parameter_bytes(full)
    assert model.resident_parameter_bytes(full) == model.total_parameter_bytes()


def test_explicit_full_plan_matches_default() -> None:
    model = make_model()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))

    with torch.no_grad():
        default_logits = model(input_ids).logits
        explicit_logits = model(input_ids, plan=PagePlan.full()).logits

    torch.testing.assert_close(default_logits, explicit_logits)

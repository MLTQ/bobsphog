import pytest
import torch

from bobsphog.catalog import PageCatalog
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.physical_cache import PhysicalPageCache


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_physical_cache_matches_resident_execution_and_reuses_pages() -> None:
    torch.manual_seed(16)
    device = torch.device("cuda:0")
    model = ToyTransformer(
        ToyConfig(
            vocab_size=16,
            context_length=8,
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            base_rank=4,
            page_rank=4,
            factorized_page_count=3,
        )
    ).eval().to(device)
    catalog = PageCatalog.from_model(model)
    selected = catalog.static_prefix(2)
    plan = catalog.plan(selected)
    inputs = torch.randint(0, 16, (2, 8), device=device)
    with torch.inference_mode():
        reference = model(inputs, plan=plan).logits

    cache = PhysicalPageCache(
        model,
        device=device,
        capacity_bytes=catalog.selected_bytes(selected),
        dtype=torch.float32,
    )
    cache.schedule(plan)
    with torch.inference_mode():
        physical = model(inputs, plan=plan).logits
    torch.cuda.synchronize(device)
    torch.testing.assert_close(physical, reference)
    cold = cache.snapshot()
    cache.prepare(plan)
    warm = cache.snapshot()

    assert cold.misses == 2
    assert cold.host_wait_seconds == 0
    assert warm.hits - cold.hits == 2
    assert cache.cache_bytes == catalog.selected_bytes(selected)
    assert all(
        parameter.device.type == "cpu"
        for layer in model.paged_layers().values()
        for page in layer.pages
        for parameter in page.parameters()
    )

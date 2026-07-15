import torch

from bobsphog.catalog import PageCatalog
from bobsphog.compositional import CompositionalArithmetic
from bobsphog.model import ToyConfig, ToyTransformer
from bobsphog.relationship import build_relationship_graph


def test_compositional_answers_and_relationship_graph() -> None:
    torch.manual_seed(15)
    task = CompositionalArithmetic(context_length=11, base=8)
    batch = task.sample(
        8,
        generator=torch.Generator().manual_seed(1),
        domain="add_then_multiply",
    )
    answer_indices = batch.answer_mask[0].nonzero().flatten().tolist()
    assert answer_indices == [5, 10]
    for answer_index in answer_indices:
        left = batch.input_ids[0, answer_index - 3] - task.NUMBER_OFFSET
        middle = batch.input_ids[0, answer_index - 2] - task.NUMBER_OFFSET
        right = batch.input_ids[0, answer_index - 1] - task.NUMBER_OFFSET
        expected = ((left + middle) * right) % task.base + task.NUMBER_OFFSET
        assert batch.targets[0, answer_index] == expected

    model = ToyTransformer(
        ToyConfig(
            vocab_size=task.vocab_size,
            context_length=11,
            d_model=8,
            n_heads=2,
            n_layers=1,
            d_ff=16,
            base_rank=2,
            page_rank=2,
        )
    ).eval()
    catalog = PageCatalog.from_model(model)
    graph = build_relationship_graph(
        model,
        batch,
        catalog,
        candidate_pool=4,
        neighbors_per_page=2,
    )

    assert graph.page_count == len(catalog)
    assert len(graph.singleton_ranking(2)) == 2
    assert len(graph.graph_greedy_selection(2)) == 2
    assert all(weight != 0 for weight in graph.edges.values())

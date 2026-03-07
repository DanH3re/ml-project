from src.pipeline.registry import registry
from src.pipeline.context import Context
from src.embedding.model import build_model


@registry.step(order=2)
def step_embedding(ctx: Context) -> None:
    ctx.model = build_model(ctx.vocab_size, ctx.d, ctx.max_seq_length)

from src.pipeline.registry import registry
from src.pipeline.context import Context
from src.data.datasetsss import load_brown, load_ud


@registry.step(order=1)
def step_data(ctx: Context) -> None:
    if ctx.use_brown:
        preprocessed, vocab, encoded = load_brown()
    else:
        preprocessed, vocab, encoded = load_ud()

    ctx.preprocessed = preprocessed
    ctx.vocab = vocab
    ctx.encoded = encoded
    ctx.vocab_size = len(vocab.word2id)

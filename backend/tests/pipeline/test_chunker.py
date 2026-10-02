from modules.pipeline.adapters.translation.chunker import Chunker
from modules.pipeline.domain.models import TextBlock


def test_chunker_empty_input():
    chunker = Chunker()
    assert chunker.chunk(()) == []


def test_chunker_single_block():
    chunker = Chunker(max_chars=100)
    block = TextBlock(order=1, kind="paragraph", text="Habari za asubuhi")
    chunks = chunker.chunk((block,))
    assert len(chunks) == 1
    assert chunks[0].text == "Habari za asubuhi"
    assert chunks[0].block_orders == (1,)


def test_chunker_multiple_blocks():
    chunker = Chunker(max_chars=100)
    block1 = TextBlock(order=1, kind="paragraph", text="Kipande cha kwanza.")
    block2 = TextBlock(order=2, kind="paragraph", text="Kipande cha pili.")
    chunks = chunker.chunk((block1, block2))
    assert len(chunks) >= 1
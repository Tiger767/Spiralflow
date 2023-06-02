import pytest
from unittest.mock import MagicMock
from spiralflow.chunking import Chunker, SmartChunker


# TestChunker
class TestChunker:
    def test_init(self):
        encoder = MagicMock()
        chunker = Chunker(encoder, 100, 0.2)
        assert chunker.encoder == encoder
        assert chunker.chunk_size == 100
        assert chunker.overlap_factor == 0.2

    def test_chunk(self):
        encoder = MagicMock()
        encoder.encode.side_effect = lambda text: list(text)
        encoder.decode.side_effect = lambda tokens: "".join(tokens)

        chunker = Chunker(encoder, 5, 0.2)
        text = "This is a sample text for testing."
        result = chunker.chunk(text)

        expected_result = [
            "This ",
            "s is ",
            "s a s",
            "ample",
            "e tex",
            "t for",
            " test",
            "ting.",
        ]
        assert result == expected_result


# TestSmartChunker
class TestSmartChunker:
    def test_init(self):
        # Test with default parameters
        encoder = MagicMock()
        smart_chunker = SmartChunker(encoder, 100, 0.2)
        assert smart_chunker.encoder == encoder
        assert smart_chunker.chunk_size == 100
        assert smart_chunker.overlap_factor == 0.2

        # Test with custom parameters
        custom_delimiters = [(" ", 0.5, True)]
        smart_chunker = SmartChunker(encoder, 100, 0.2, custom_delimiters)
        assert smart_chunker._delimiters_tolerances_overlap == custom_delimiters

    def test_chunk(self):
        encoder = MagicMock()
        encoder.encode.side_effect = lambda text: list(text)
        encoder.decode.side_effect = lambda tokens: "".join(tokens)

        smart_chunker = SmartChunker(encoder, 7, 0.2)
        text = "This is a sample text for testing."
        result = smart_chunker.chunk(text)

        expected_result = [
            "This is",
            "is a sa",
            "a sample",
            "sample text",
            " text for",
            "for testing.",
        ]
        assert result == expected_result

        # Test with custom delimiter, tolerance, and overlap
        custom_delimiters = [(",", 0.5, True)]
        smart_chunker = SmartChunker(encoder, 7, 0.2, custom_delimiters)
        text = "This is,a sample,text for,testing."
        result = smart_chunker.chunk(text)

        expected_result = [
            "This is",
            "is,a sam",
            "a sample",
            "sample,text for",
            "text for",
            "for,testing.",
        ]
        assert result == expected_result

        # Test with prefer_large_chunks
        smart_chunker = SmartChunker(encoder, 7, 0.2, prefer_large_chunks=False)
        text = "This is a sample text for testing."
        result = smart_chunker.chunk(text)

from typing import List, Tuple, Optional


class Chunker:
    def __init__(self, encoder, chunk_size: int, overlap_factor: float) -> None:
        """
        :param encoder: The text encoder object.
        :param chunk_size: The desired chunk size in token units.
        :param overlap_factor: The factor to calculate the overlap between chunks.
        """
        self.encoder = encoder
        self.chunk_size = chunk_size
        self.overlap_factor = overlap_factor

    def chunk(self, text: str) -> List[str]:
        """
        Chunks text into chunks of size chunk_size (token units) with overlap.

        :param text: The input text to be chunked.
        :returns: A list of chunked text.
        """
        overlap = int(self.chunk_size * self.overlap_factor)
        tokens = self.encoder.encode(text)

        chunked_tokens = [
            tokens[i : i + self.chunk_size]
            for i in range(0, len(tokens), self.chunk_size - overlap)
        ]

        return [self.encoder.decode(chunk) for chunk in chunked_tokens]


class SmartChunker(Chunker):
    def __init__(
        self,
        encoder,
        chunk_size: int,
        overlap_factor: float,
        delimiters_tolerances_overlap: Optional[List[Tuple[str, float, bool]]] = None,
        prefer_large_chunks: bool = True,
    ) -> None:
        """
        :param encoder: The text encoder object.
        :param chunk_size: The desired chunk size in token units.
        :param overlap_factor: The factor to calculate the overlap between chunks.
        :param delimiters_tolerances_overlap: A list of tuples with delimiter,
            tolerance, and overlap values for smart chunking. Defaults to None.
        :param prefer_large_chunks: If True, prefer chunks splitting at last occurance of a delimiter.
        """
        super().__init__(encoder, chunk_size, overlap_factor)
        self._delimiters_tolerances_overlap = (
            [
                ("\n\n\n", 0.4, False),
                ("\n\n", 0.2, False),
                ("\n", 0.1, True),
                (" ", 0.5, True),
                ("", 0.01, True),
            ]
            if delimiters_tolerances_overlap is None
            else delimiters_tolerances_overlap
        )
        self.prefer_large_chunks = prefer_large_chunks

        # should check that int(self.overlap_factor * chunk_size) is always less than chunk_size * smallest possible tolerance

    def chunk(self, text: str) -> List[str]:
        """
        Chunks text respecting delimiters, tolerances, and overlap into chunk_size with overlap.

        :param text: The input text to be chunked.
        :returns: A list of chunked text.
        """
        chunk_size = self.chunk_size

        final_chunks = []

        tokens = self.encoder.encode(text)

        start_ndx = 0
        while start_ndx + chunk_size < len(tokens):
            best_chunk_end_ndx = start_ndx

            for delimiter, tolerance, overlap in self._delimiters_tolerances_overlap:
                search_start = start_ndx + int(chunk_size * (1 - tolerance))
                search_end = start_ndx + chunk_size

                subtext = self.encoder.decode(tokens[search_start:search_end])
                if self.prefer_large_chunks:
                    delimiter_ndx = subtext.rfind(delimiter)
                else:
                    delimiter_ndx = subtext.find(delimiter)

                if delimiter_ndx != -1:
                    delimiter_ndx = len(self.encoder.encode(subtext[:delimiter_ndx]))
                    best_chunk_end_ndx = delimiter_ndx + search_start
                    break

            final_chunks.append(self.encoder.decode(tokens[start_ndx:best_chunk_end_ndx]))

            start_ndx = best_chunk_end_ndx
            if overlap:
                overlap_offset = int(self.overlap_factor * chunk_size)
                ndx = text[start_ndx - overlap_offset : start_ndx].find(" ")
                if ndx != -1:
                    start_ndx += ndx - overlap_offset
        # if start_ndx < len(text):
        final_chunks.append(self.encoder.decode(tokens[start_ndx:]))

        return final_chunks

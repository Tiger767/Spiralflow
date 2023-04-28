from typing import List, Tuple, Union


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
        delimiters_tolerances_overlap: Union[
            None, List[Tuple[str, float, bool]]
        ] = None,
    ) -> None:
        """
        :param encoder: The text encoder object.
        :param chunk_size: The desired chunk size in token units.
        :param overlap_factor: The factor to calculate the overlap between chunks.
        :param delimiters_tolerances_overlap: A list of tuples with delimiter,
            tolerance, and overlap values for smart chunking. Defaults to None.
        """
        super().__init__(encoder, chunk_size, overlap_factor)

        self.delimiters_tolerances_overlap = (
            [
                ("\n\n\n", 0.3, False),
                ("\n\n", 0.15, False),
                ("\n", 0.05, True),
                (" ", 0.5, True),
                ("", 0.01, True),
            ]
            if delimiters_tolerances_overlap is None
            else delimiters_tolerances_overlap
        )

    def chunk(self, text: str) -> List[str]:
        """
        Chunks text respecting delimiters, tolerances, and overlap into chunk_size
        (estimated token units) with overlap.

        :param text: The input text to be chunked.
        :returns: A list of chunked text.
        """
        token_chunk_size = self.chunk_size
        chunk_size = self.chunk_size * 4  # estimate token size

        final_chunks = []

        start_ndx = 0
        while start_ndx + chunk_size < len(text):
            best_chunk_end_ndx = start_ndx

            for delimiter, tolerance, overlap in self.delimiters_tolerances_overlap:
                search_start = start_ndx + int(chunk_size * (1 - tolerance))
                search_end = start_ndx + chunk_size
                delimiter_ndx = text[search_start:search_end].rfind(delimiter)

                if delimiter_ndx != -1:
                    best_chunk_end_ndx = delimiter_ndx + search_start
                    break

            chunk = text[start_ndx:best_chunk_end_ndx]

            # Make sure chunk is actually smaller than chunk_size after the fact
            # may not lead to best results if truncation is needed
            encoded_chunk = self.encoder.encode(chunk)
            if len(encoded_chunk) > token_chunk_size:
                print("chunk too large, truncating")
                encoded_chunk = encoded_chunk[:token_chunk_size]
                trunc_chunk = self.encoder.decode(encoded_chunk)
                best_chunk_end_ndx -= len(chunk) - len(trunc_chunk)
                chunk = trunc_chunk

            final_chunks.append(chunk)

            start_ndx = best_chunk_end_ndx
            if overlap:
                overlap_offset = int(self.overlap_factor * chunk_size)
                ndx = text[start_ndx - overlap_offset : start_ndx].find(" ")
                if ndx != -1:
                    start_ndx += ndx - overlap_offset
        # if start_ndx < len(text):
        final_chunks.append(text[start_ndx:])

        return final_chunks

import numpy as np
import pandas as pd
import faiss
from typing import Dict, Optional
import pickle
import tiktoken
from openai.embeddings_utils import get_embedding


class Memory:
    def __init__(
        self,
        filepath: Optional[str] = None,
        embedding_model: str = "text-embedding-ada-002",
        max_tokens: int = 500,
    ) -> None:
        """
        Initializes the memory.

        :param filepath: Path to a pickle file to load and save the memory to.
                         If None, the memory is created with text and metadata fields.
        :param embedding_model: Model to use for the embedding.
        :param max_tokens: Maximum number of tokens to use for the embedding.
        """
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(self.embedding_model)

        if filepath is None:
            self.data = pd.DataFrame(columns=["text", "metadata"])
            self.index = None
        else:
            self.filepath = filepath
            self.load()

    def _create_index(self, embeddings: np.ndarray) -> None:
        _, d = embeddings.shape
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings)

    def save(self, filepath: Optional[str] = None) -> None:
        """
        Saves the memory to a file.

        :param filepath: Path to the pickle file to save the memory to. If None, the filepath passed in the constructor is used.
        """
        if filepath is None:
            filepath = self.filepath

        self.encoding, encoding = None, self.encoding
        with open(filepath + ".pkl", "wb") as f:
            pickle.dump(self, f)
        self.encoding = encoding

    def load(self, filepath: Optional[str] = None) -> None:
        """
        :param filepath: Path to a pickle file to load the memory from. If None, the filepath passed in the constructor is used.
        """
        if filepath is None:
            filepath = self.filepath

        with open(filepath, "rb") as f:
            loaded_memory = pickle.load(f)
            self.data = loaded_memory.data
            self.index = loaded_memory.index
            self.embedding_model = loaded_memory.embedding_model
            self.max_tokens = loaded_memory.max_tokens
            self.encoding = tiktoken.encoding_for_model(self.embedding_model)

    def add(
        self, data: Dict[str, str], save: bool = False, filepath: Optional[str] = None
    ) -> None:
        """
        Adds data to memory.

        :param data: Dict of data with a text and metadata field to add to memory.
        :param save: Whether to save the memory to a file.
        :param filepath: Path to the file (csv or parquet) to save the memory to.
                         If None, the filepath passed in the constructor is used.
        """

        if "text" not in data:
            raise ValueError("Data must have a 'text' field.")

        if len(self.encoding.encode(data["text"])) > self.max_tokens:
            raise ValueError(
                "Text must be less than {} tokens.".format(self.max_tokens)
            )

        # get embedding of text
        embedding = np.array(
            [get_embedding(data["text"], engine=self.embedding_model)], dtype=np.float32
        )

        data = pd.DataFrame(data, index=[0])
        self.data = pd.concat([self.data, data], ignore_index=True)

        if self.index is None:
            self._create_index(embedding)
        else:
            self.index.add(embedding)

        if save:
            self.save(filepath)

    def query(self, query: str, k: int = 1) -> list[Dict[str, str]]:
        """
        Queries the memory with the given query.

        :param query: Query to use to get memory.
        :param k: Number of results to return.
        :return: Memory obtained from external memories.
        """
        if self.data.empty:
            raise ValueError(
                "No memory to query. Add data to memory by calling Memory.add() before querying."
            )

        if len(self.encoding.encode(query)) > self.max_tokens:
            raise ValueError(
                "Text must be less than {} tokens.".format(self.max_tokens)
            )

        # get embedding of query
        embeded_query = np.array([get_embedding(query, engine=self.embedding_model)])

        # search for the indexes with similar embeddings
        _, similar_indexes = self.index.search(embeded_query, k=k)

        # get the memory values at the found indexes
        memories = []

        for i in similar_indexes[0]:
            memory = {"text": self.data.iloc[i, 0], "metadata": self.data.iloc[i, 1]}
            memories.append(memory)

        return memories

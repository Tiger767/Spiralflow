import numpy as np
import pandas as pd
import faiss
from typing import Dict, Optional
import pickle
import tiktoken
from openai.embeddings_utils import get_embedding


def get_percent_similar(str1: str, str2: str) -> float:
    """
    Calculates the percentage of similarity between two strings using levenshtein string comparison.

    :param str1: First string.
    :param str2: Second string.
    :return: Percentage of similarity between two strings represented by a float.
    """
    # initialize levenshtein distance matrix
    matrix = np.zeros((len(str1) +1, len(str2)+1))
    for i in range(len(str1)+1):
        matrix[i][0] = i
    for i in range(len(str2)+1):
        matrix[0][i] = i
    # calculate levenshtein distance
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            matrix[i][j] = min(matrix[i-1][j], matrix[i][j-1], matrix[i-1][j-1])
            if str1[i-1] != str2[j-1]:
                matrix[i][j] = matrix[i][j]+1

    # The distance is the last item in the matrix. Use this to calculate a percentage match
    distance = matrix[len(str1)][len(str2)]
    bigger = max(len(str1), len(str2))
    return (bigger - distance) / bigger

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

    def query(
        self, query: str, k: int = 1, combine_threshold: Optional[float] = None
    ) -> list[Dict[str, str]]:
        """
        Queries the memory with the given query.

        :param query: Query to use to get memory.
        :param k: Max number of results to return.
        :param combine_threshold: A ratio representing how similar two memories can be before combining them.
                                  Two memories must be at least combine_threshold similar to be combined.
                                  In the case of combination, the memory most closely matching the query is returned.
                                  If None, no combining is done.
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
        scores, similar_indexes = self.index.search(embeded_query, k=k)

        print(scores)

        # get the memory values at the found indexes
        memories = []

        for score, memA in zip(scores[0], similar_indexes[0]):
            memory = {
                "text": self.data.iloc[memA, 0],
                "metadata": self.data.iloc[memA, 1],
                "score": score,
            }
            memories.append(memory)
        
        if combine_threshold is not None and len(memories) > 1:
            combined_memories = []
            for memA in memories:
                # create a group of memories that are similar enough to memA to be combined
                combine_group = [memA]
                for memB in memories:
                    if memA != memB:
                        # get the percentage similarity between memory a and memory b
                        a_to_b_similarity = get_percent_similar(memA["text"], memB["text"])
                        # if this is greater than the threshold add memB to the combine group
                        if a_to_b_similarity >= combine_threshold:
                            combine_group.append(memB)
                # get the memory with the highest similarity to the query in the group
                most_similar = (0, None)
                for mem in combine_group:
                    mem_sim = get_percent_similar(mem["text"], query)
                    if mem_sim > most_similar[0]:
                        most_similar = (mem_sim, mem)
                # do not add a duplicate to the combined memories
                if most_similar[1] not in combined_memories:
                    combined_memories.append(most_similar[1])
            memories = combined_memories
        return memories

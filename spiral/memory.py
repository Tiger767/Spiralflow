import faiss
from openai.embeddings_utils import get_embedding, cosine_similarity


class Memory:
    def __init__(self):
        pass

    def query(self, query: str, k=1) -> str:
        """
        Queries the memory with the given query.

        :param query: Query to use to get memory.
        :param k: Number of results to return.
        :return: Memory obtained from external memories.
        """
        raise NotImplementedError("")

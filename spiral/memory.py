import faiss
import numpy as np
import pandas as pd
import ast
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

        # get embedding of query
        xQuery = np.array([get_embedding(query, engine="text-embedding-ada-002")])

        # get vectors from memory 
        df = pd.read_csv("data/memory.csv", index_col=0)
        vectors = []

        for val in df[["embedding"]].values:
            vectors.append(ast.literal_eval(val[0]))

        vectors = np.array(vectors)

        # get faiss index to utilize search
        _, d = vectors.shape
        index = faiss.IndexFlatL2(d)
        index.add(vectors)

        # search for the indexes with similar embeddings
        D, similarIndexes = index.search(xQuery, k)

        # get the memory values at the found indexes
        memoryValues = []
        for i in similarIndexes[0]:
            memoryValues.append(df[["combined"]].values[i])
        return [memoryValues , D]
        
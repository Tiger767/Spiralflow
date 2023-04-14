import faiss
import numpy as np
import pandas as pd
import ast
from openai.embeddings_utils import get_embedding, cosine_similarity
import tiktoken



class Memory:
    def __init__(self, filepath = "data/memory.csv",  max_tokens = 500, embedding_model = "text-embedding-ada-002"):
        self.filepath = filepath
        self.data = self.load(filepath) 
        self.embedding_model = embedding_model
        self.max_tokens = max_tokens
        # self.embedding_encoding = "cl100k_base"
        pass

    def save(self, filepath):
        self.data.to_csv(filepath)

    def load(self, filepath):
        try:
            df = pd.read_csv(filepath, index_col=0)
        except:
            df = pd.DataFrame(columns=["embedding", "text", "metadata"])
        return df

    def add(self, data: object, save: bool = False):
        """
        Adds data to memory.

        :param data: Array of data with a text and metadata field to add to memory.
        :param save: Whether to save the memory to a file.
        """

        if "text" not in data:
            raise ValueError("Data must have a 'text' field.")

        # get embedding of text
        embedding = get_embedding(data["text"], engine=self.embedding_model)
        data["embedding"] = [ embedding ]
        
        data = pd.DataFrame(data)
        self.data = pd.concat([self.data, data], ignore_index=True)

        if save:
            self.save(self.filepath)

    def query(self, query: str, k=1) -> str:
        """
        Queries the memory with the given query.

        :param query: Query to use to get memory.
        :param k: Number of results to return.
        :return: Memory obtained from external memories.
        """
        if self.data.empty:
            raise ValueError("No memory to query. Add data to memory by calling Memory.add() before querying.")

        # get vectors from memory 
        vectors = []
        for val in self.data[["embedding"]].values:
            if type(val[0]) == str:
                vectors.append(ast.literal_eval(val[0]))
            else:
                vectors.append(val[0])

        vectors = np.array(vectors)

        # get faiss index to utilize search
        _, d = vectors.shape
        index = faiss.IndexFlatL2(d)
        index.add(vectors)

        # get embedding of query
        xQuery = np.array([get_embedding(query, engine=self.embedding_model)])

        # search for the indexes with similar embeddings
        _, similarIndexes = index.search(xQuery, k)

        # get the memory values at the found indexes
        memory = {
            "text":"",
            "metadata":""
        }
        memories = []

        for i in similarIndexes[0]:
            memory["text"] = self.data.iloc[i, 1]
            memory["metadata"] = self.data.iloc[i, 2]
            memories.append(memory.copy())

        return memories



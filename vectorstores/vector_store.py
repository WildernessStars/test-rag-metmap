import numpy as np
from langchain_community.vectorstores import Annoy
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import ScaNN
from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings, HuggingFaceEmbeddings


# model_name = "sentence-transformers/all-mpnet-base-v2"
# embeddings_func = HuggingFaceEmbeddings(model_name=model_name)


class VectorDatabase:
    def __init__(self, texts, embeddings_func, database_type='Annoy'):
        self.db = self.initialize_database(database_type, texts, embeddings_func)

    def initialize_database(self, database_type, texts, embeddings_func):
        if database_type == 'Annoy':
            # sentence-transformers/all-mpnet-base-v2
            return Annoy.from_texts(texts, embeddings_func)
        elif database_type == 'Chroma':
            # all-MiniLM-L6-v2
            return Chroma.from_texts(texts, embeddings_func)
        elif database_type == 'ScaNN':
            # Placeholder for ScaNN initialization
            return ScaNN.from_texts(texts, embeddings_func)

    def simulate_retrieval(self, query):
        # Simulate retrieval by finding the most similar vector
        return self.db.similarity_search(query, k=1)

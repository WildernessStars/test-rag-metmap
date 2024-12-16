
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from llama_index.readers.huggingface_fs import HuggingFaceFSReader
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
from llama_index.core.text_splitter import SentenceSplitter
from llama_index.core import (
    SimpleDirectoryReader,
    load_index_from_storage,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings


def filter_html(tokens: np.array, is_html: np.array, start: int = 0, end: int = None):
    if end is None:
        end = len(tokens)
    end = min(end, len(tokens))
    return tokens[start: end][~is_html[start: end]]


def extract_text_from_html(html_content):
    soup = BeautifulSoup(html_content, "html.parser")
    text = soup.get_text(separator=" ")
    text = ' '.join(text.split())

    return text


def load_dataset(path: str):
    df = pd.read_parquet(path)
    annotation = df.iloc[0, 4]
    tokens = df.iloc[0, 1]['tokens']['token']
    is_html = df.iloc[0, 1]['tokens']['is_html']
    context = df.iloc[:, 1].apply(lambda x: extract_text_from_html(x['html']))
    question = df.iloc[:, 2].apply(lambda x: x['text'])
    print(question.head(5))

    def extract_answer(content, annotation, type):
        tokens = content['token']
        is_html = content['is_html']
        start_token_long = annotation[type][0]['start_token']
        end_token_long = annotation[type][0]['end_token']
        if type == 'short_answers':
            start_token_long = start_token_long[0]
            end_token_long = end_token_long[0]
        return ' '.join(filter_html(tokens, is_html, start_token_long, end_token_long))

    long_answer = df.apply(lambda r: extract_answer(r[1]['tokens'], r[4], 'long_answer'), axis=1)
    print(long_answer.head(5))
    # print(long_answer.iloc[2, :])
    short_answer = df.apply(lambda r: extract_answer(r[1]['tokens'], r[4], 'short_answers'), axis=1)
    print(short_answer.head(5))
    # candidates = df.iloc[0, 3]
    # print(candidates)
    # candidates_end = candidates['end_token']
    # candidates_start = candidates['start_token']
    # candidate_answers = [' '.join(filter_html(tokens, is_html, s, e)) for s, e in zip(candidates_start, candidates_end)]
    # print(candidate_answers[0:5])


def get_question():
    pass


def evaluate_retrieval_model():
    # "google-research-datasets/natural_questions"
    # loader = HuggingFaceFSReader()
    # documents = loader.load_data("datasets/dair-ai/emotion/data/data.jsonl.gz")
    load_dataset('./data/natural_questions/train-00000-of-00287.parquet')

    # create a faiss index
    d = 1536  # dimension
    faiss_index = faiss.IndexFlatL2(d)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    documents = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
    print(documents)
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    Settings.embed_model = embed_model
    parser = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = parser(documents)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )
    index.storage_context.persist()
    VectorStoreIndex(nodes, storage_context=storage_context)
    query_engine = index.as_query_engine()
    response = query_engine.query("What did the author do growing up?")
    metrics_k_values = [3, 10]


evaluate_retrieval_model()

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from llama_index.core import Settings
import logging
from datetime import datetime
import sys

logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.handlers = []
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)  # Set handler level to INFO
logger.addHandler(handler)


class MyAzureOpenAI(AzureOpenAI):
    def __init__(self, engine="gpt-35-turbo"):
        super().__init__(engine=engine,
                         model="gpt-4o",
                         api_key="",
                         api_version="2024-06-01",
                         azure_endpoint="https://hkust.azure-api.net"
                         )

    def chat(self, messages, **kwargs):
        logger.debug('chat')
        logger.debug(messages)
        logger.debug(self.metadata)
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.engine,
            n=1,
            messages=messages
        )
        return response.choices[0]


class MyAzureOpenAIEmbedding(AzureOpenAIEmbedding):
    def __init__(self):
        super().__init__(model="text-embedding-ada-002",
                         api_key="",
                         api_version="2024-06-01",
                         azure_endpoint="https://hkust.azure-api.net",
                         )

    def _get_text_embedding(self, text):
        client = self._get_client()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0]


embed_model = MyAzureOpenAIEmbedding()

Settings.embed_model = embed_model
data = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()
index = VectorStoreIndex.from_documents(data)

Settings.llm = MyAzureOpenAI(engine="gpt-35-turbo")
print(datetime.now())

# Settings.llm = CustomAzureOpenAILLM()
query = "What are the first programs Paul Graham tried writing?"
query_engine = index.as_query_engine()
response = query_engine.query(query)
print(response)
print(datetime.now())

from __future__ import annotations

import asyncio
import typing as t
from abc import ABC
from dataclasses import field
from typing import List, Any, Optional

import torch.cuda
from langchain_cohere import CohereEmbeddings
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from pydantic.dataclasses import dataclass
from run_config import RunConfig, add_async_retry, add_retry
from torch.nn.functional import normalize
import paddle

if t.TYPE_CHECKING:
    from llama_index.core.base.embeddings.base import BaseEmbedding

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_PADDLE_MODEL_NAME = "PaddlePaddle/ernie-3.0-medium-zh"
DEFAULT_RWKV_MODEL_NAME = "sgugger/rwkv-430M-pile"


class BaseEmbeddings(Embeddings, ABC):
    run_config: RunConfig

    async def embed_text(self, text: str, is_async=True) -> List[float]:
        """
        Embed a single text string.
        """
        embs = await self.embed_texts([text], is_async=is_async)
        return embs[0]

    async def embed_texts(
            self, texts: List[str], is_async: bool = True
    ) -> t.List[t.List[float]]:
        if is_async:
            aembed_documents_with_retry = add_async_retry(
                self.aembed_documents, self.run_config
            )
            return await aembed_documents_with_retry(texts)
        else:
            loop = asyncio.get_event_loop()
            embed_documents_with_retry = add_retry(
                self.embed_documents, self.run_config
            )
            return await loop.run_in_executor(None, embed_documents_with_retry, texts)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config


class LangchainEmbeddingsWrapper(BaseEmbeddings):
    def __init__(
            self, embeddings: Embeddings, run_config: t.Optional[RunConfig] = None
    ):
        self.embeddings = embeddings
        if run_config is None:
            run_config = RunConfig()
        self.set_run_config(run_config)

    def embed_query(self, text: str) -> List[float]:
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embeddings.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.embeddings.aembed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.embeddings.aembed_documents(texts)

    def set_run_config(self, run_config: RunConfig):
        self.run_config = run_config

        # run configurations specially for OpenAI
        if isinstance(self.embeddings, OpenAIEmbeddings):
            try:
                from openai import RateLimitError
            except ImportError:
                raise ImportError(
                    "openai.error.RateLimitError not found. Please install openai package as `pip install openai`"
                )
            self.embeddings.request_timeout = run_config.timeout
            self.run_config.exception_types = RateLimitError


@dataclass
class HuggingfaceEmbeddings(BaseEmbeddings):
    model_name: str = DEFAULT_MODEL_NAME
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import sentence_transformers
            from transformers import AutoConfig, AutoModel, AutoTokenizer, \
                RwkvModel, AutoModelForCausalLM, BitsAndBytesConfig, DistilBertTokenizer, \
                DistilBertModel  # , , GPTQConfig
            from transformers import XLNetTokenizer, XLNetModel
            from transformers.models.auto.modeling_auto import (
                MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            )
        except ImportError as exc:
            raise ImportError(
                "Could not import sentence_transformers python package. "
                "Please install it with `pip install sentence-transformers`."
            ) from exc
        config = AutoConfig.from_pretrained(self.model_name)
        self.is_cross_encoder = bool(
            np.intersect1d(
                list(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES.values()),
                config.architectures if config.architectures is not None else '',
            )
        )
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.is_cross_encoder:
            self.model = sentence_transformers.CrossEncoder(
                self.model_name, device=device, **self.model_kwargs
            )
        else:
            # gptq_config = GPTQConfig(bits=4)
            # self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
            #                                                   quantization_config=gptq_config,
            #
            if self.model_name in ['SpanBERT/spanbert-large-cased',
                                   'google/electra-large-generator',
                                   'google-bert/bert-base-uncased',
                                   'michiyasunaga/LinkBERT-base',
                                   'FacebookAI/roberta-base',
                                   'albert/albert-base-v2',
                                   'sentence-transformers/gtr-t5-large',
                                   'sentence-transformers/sentence-t5-large',
                                   'sentence-transformers/all-mpnet-base-v2',
                                   'sentence-transformers/all-MiniLM-L6-v2',
                                   ]:
                self.model = sentence_transformers.SentenceTransformer(
                    self.model_name, cache_folder=self.cache_folder, **self.model_kwargs
                )
                if "convert_to_tensor" not in self.encode_kwargs:
                    self.encode_kwargs["convert_to_tensor"] = True
            elif self.model_name == 'xlnet/xlnet-large-cased':
                self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
                self.model = XLNetModel.from_pretrained('xlnet-large-cased')
            elif self.model_name == 'sgugger/rwkv-430M-pile':
                self.model = RwkvModel.from_pretrained(self.model_name, output_hidden_states=False,
                                                       output_attentions=False)
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            elif 'Q4' in self.model_name:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, gguf_file='falcon-7b-q4_0.gguf',
                                                                  device_map="cuda:0")
                self.model.eval()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, gguf_file='falcon-7b-q4_0.gguf')
            elif self.model_name == 'distilbert-base-uncased':
                print('ddddd')
                from transformers import DistilBertTokenizer, DistilBertModel
                self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
                self.model = DistilBertModel.from_pretrained("distilbert-base-uncased")
            else:
                if '4bit' in self.model_name or 'load_in_4bit' in self.model_kwargs:
                    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                                      quantization_config=quantization_config)
                    if '4bit' not in self.model_name:
                        self.model_name = self.model_name + '-4bit'
                else:
                    self.model = AutoModel.from_pretrained(self.model_name, **self.model_kwargs)
                if 'llama' in self.model_name:
                    self.tokenizer = AutoTokenizer.from_pretrained('hf-internal-testing/llama-tokenizer')
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            self.device = device
            self.model.to(device)

    def embed_query(self, text: str) -> List[float]:
        if 'llama' in self.model_name or 'Llama' in self.model_name or 'falcon-7b' in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            seq_ids = self.tokenizer(text, return_tensors='pt')["input_ids"].to(self.device)
            return self.model(seq_ids, output_hidden_states=True).hidden_states[-1].mean(
                axis=[0, 1]).detach().cpu().data.numpy()
            # return self.model(seq_ids, output_hidden_states=True)["last_hidden_state"].mean(axis=[0, 1]).detach().cpu().data.numpy()
        if 'rwkv' in self.model_name:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                emb = outputs.last_hidden_state[0, 0, :].detach().cpu().data
                del inputs, outputs
                torch.cuda.empty_cache()
                return emb
        if 'xlnet' in self.model_name:
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            last_hidden_states = outputs.last_hidden_state
            return last_hidden_states.mean(axis=[0, 1]).detach().cpu().data
        if self.model_name == 'distilbert-base-uncased':
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**encoded_input)
                token_embeddings = outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
                attention_mask = encoded_input['attention_mask']  # Shape: (batch_size, seq_len)
                attention_mask = attention_mask.unsqueeze(-1).expand(
                    token_embeddings.size())  # Shape: (batch_size, seq_len, hidden_size)
                masked_embeddings = token_embeddings * attention_mask
                sum_embeddings = torch.sum(masked_embeddings, dim=1)
                sum_mask = torch.sum(attention_mask, dim=1)
                sentence_embeddings = sum_embeddings / sum_mask
                sentence_embeddings = normalize(sentence_embeddings, p=2, dim=1)  #
                return sentence_embeddings.detach().cpu().data
        return self.embed_documents([text])[0]

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        import torch
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        from sentence_transformers.SentenceTransformer import SentenceTransformer
        from torch import Tensor

        assert isinstance(
            self.model, SentenceTransformer
        ), "Model is not of the type Bi-encoder"
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, **self.encode_kwargs
        )

        assert isinstance(embeddings, Tensor)
        return embeddings.tolist()

    def predict(self, texts: List[List[str]]) -> List[List[float]]:
        """
        Make predictions using a cross-encoder model.
        """
        from sentence_transformers.cross_encoder import CrossEncoder

        assert isinstance(
            self.model, CrossEncoder
        ), "Model is not of the type CrossEncoder"

        predictions = self.model.predict(texts, **self.encode_kwargs)
        return predictions.tolist()


@dataclass
class PaddlePaddleEmbeddings(BaseEmbeddings):
    model_name: str = DEFAULT_PADDLE_MODEL_NAME
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            from paddlenlp.transformers import ErnieTokenizer, ErnieModel
        except ImportError as exc:
            raise ImportError(
                "Could not import transformers python package. "
                "Please install it with `pip install transformers`."
            ) from exc

        self.tokenizer = ErnieTokenizer.from_pretrained(self.model_name, from_hf_hub=True)
        self.model = ErnieModel.from_pretrained(self.model_name, from_hf_hub=True)
        self.model.eval()
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.model.to(device)

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pd")
        with paddle.no_grad():
            sequence_output, pooled_output = self.model(**inputs)
        embedding = sequence_output.mean(axis=[0, 1])
        # embedding = paddle.nn.functional.normalize(embedding, p=2)
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return []

@dataclass
class UformEmbeddings(BaseEmbeddings):
    model_name: str = DEFAULT_PADDLE_MODEL_NAME
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            from uform import get_model, Modality
        except ImportError as exc:
            raise ImportError(
                "Could not import uform python package. "
                "Please install it with `pip install uform[torch]`."
            ) from exc

        processors, models = get_model('unum-cloud/uform3-image-text-english-base')

        model_text = models[Modality.TEXT_ENCODER]
        processor_text = processors[Modality.TEXT_ENCODER]
        text_data = processor_text('xxxx')
        self.processor = processor_text
        self.model = model_text

    def embed_query(self, text: str) -> List[float]:
        text_data = self.processor(text)
        text_embedding = self.model.encode(text_data, return_features=False)
        return text_embedding[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:

        return []


@dataclass
class OptimumEmbeddings(BaseEmbeddings):
    model_name: str = DEFAULT_PADDLE_MODEL_NAME
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Rrequires onnxruntime to be installed.\n"
            )
        onnx_model_path = hf_hub_download(repo_id=self.model_name, filename="model.onnx")
        self.model = ort.InferenceSession(onnx_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("GPTCache/paraphrase-albert-small-v2")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        import torch

        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def post_proc(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
        sentence_embs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), 1e-9
        )
        return sentence_embs

    def embed_query(self, text: str) -> List[float]:
        encoded_text = self.tokenizer.encode_plus(text, padding="max_length", return_tensors='pt')
        ort_inputs = {
            "input_ids": np.array(encoded_text["input_ids"]).astype("int64").reshape(1, -1),
            "attention_mask": np.array(encoded_text["attention_mask"]).astype("int64").reshape(1, -1),
            "token_type_ids": np.array(encoded_text["token_type_ids"]).astype("int64").reshape(1, -1),
        }
        ort_outputs = self.model.run(None, ort_inputs)
        ort_feat = torch.tensor(ort_outputs[0], device=self.device)
        emb = self.post_proc(ort_feat, torch.tensor(ort_inputs["attention_mask"], device=self.device))
        return emb.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return []


@dataclass
class FasttextEmbeddings(BaseEmbeddings):
    model_name: str = "fastText-en"
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import fasttext, fasttext.util
        except ImportError:
            raise ImportError(
                "Require fasttext to be installed."
            )
        fasttext.util.download_model('en', if_exists='ignore')
        self.model = fasttext.load_model("cc.en.300.bin")
        self.tokenizer = None

    def embed_query(self, text: str) -> List[float]:
        words = text.split()
        word_embeddings = [self.model.get_word_vector(word) for word in words]
        sentence_embedding = np.mean(word_embeddings, axis=0)
        return [float(x) for x in sentence_embedding]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return []

@dataclass
class CohereEmbeddings(BaseEmbeddings):
    model_name: str = "Cohere/embed-english-v2.0"
    cache_folder: t.Optional[str] = None
    model_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)
    encode_kwargs: t.Dict[str, t.Any] = field(default_factory=dict)

    def __post_init__(self):
        try:
            import cohere
        except ImportError:
            raise ImportError(
                "Require cohere to be installed."
            )
        cohere_test_key = ""
        cohere_key = ""
        self.model = cohere.Client(cohere_key)
        self.tokenizer = None

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        doc_emb = self.model.embed(texts=texts, input_type="search_document", model="embed-english-v2.0").embeddings
        return doc_emb


def embedding_factory(
        model: str = "text-embedding-ada-002", run_config: t.Optional[RunConfig] = None
) -> BaseEmbeddings:
    if model.endswith('ada-002'):
        openai_embeddings = OpenAIEmbeddings(model=model)
        if run_config is not None:
            openai_embeddings.request_timeout = run_config.timeout
        else:
            run_config = RunConfig()
        return LangchainEmbeddingsWrapper(openai_embeddings, run_config=run_config)
    elif model == 'embed-english-v2.0':
        cohere_embeddings = CohereEmbeddings(model=model)
        if run_config is not None:
            cohere_embeddings.request_timeout = run_config.timeout
        else:
            run_config = RunConfig()
        return LangchainEmbeddingsWrapper(cohere_embeddings, run_config=run_config)

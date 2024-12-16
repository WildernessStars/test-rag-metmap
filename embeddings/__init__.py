from embeddings.base import (
    HuggingfaceEmbeddings,
    LangchainEmbeddingsWrapper,
    CohereEmbeddings,
    PaddlePaddleEmbeddings,
    UformEmbeddings,
    OptimumEmbeddings,
    FasttextEmbeddings,
    embedding_factory,
)

from embeddings.custom import (
    CustomEmbeddings
)
__all__ = [
    "LangchainEmbeddingsWrapper",
    "CohereEmbeddings",
    "HuggingfaceEmbeddings",
    "PaddlePaddleEmbeddings",
    "UformEmbeddings",
    "OptimumEmbeddings",
    "FasttextEmbeddings",
    "embedding_factory",
    "CustomEmbeddings"
]

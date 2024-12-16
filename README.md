# test-rag-metmap

A comprehensive testing and evaluation framework for RAG (Retrieval-Augmented Generation) implementations using LangChain. According to the paper: MeTMaP: Metamorphic Testing for Detecting False Vector
Matching Problems in LLM Augmented Generation (https://arxiv.org/pdf/2402.14480)

## Overview

This project provides tools for testing and evaluating false matching problems in RAG systems with particular emphasis on embedding models in vector stores. It includes components for data processing, embedding generation, evaluation metrics, and result analysis.

## Project Structure

├── data/              # Dataset provided by paper authors  
├── embeddings/        # Generated embeddings and vector embeddings  
├── evaluate/          # examples  
├── metrics/          # Seven distance metrics  
├── results/          # Output data  
├── vectorstores/     # Vector store implementations and configs  
├── llamaindex.py     # LlamaIndex   
├── main.py           # Main application entry point  
├── qa_test.py        # Question-answering test   
├── run_config.py     # Configuration  
├── tester.py         # Testing related function  
└── word_level_tagging.py  # Word-level metamorphic relation identification  



### Evaluation Examples

- Located in the `evaluate/` directory


### Embedding Generation

- Located in the `embeddings/` directory
- Manages vector embedding of data
- Supports multiple embedding models and configurations ( HuggingfaceEmbeddings,
    LangchainEmbeddingsWrapper,
    CohereEmbeddings,
    PaddlePaddleEmbeddings,
    UformEmbeddings,
    OptimumEmbeddings,
    FasttextEmbeddings)


### Metrics Collection

- Located in the `metrics/` directory
- Supports 7 distance metrics(cosine, euclidean, person, manhattan, lancewilliams, mahalanobis, braycurtis)

### Results

- Located in the `metrics/` directory
- Including the results of evalution


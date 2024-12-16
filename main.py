from typing import List

from transformers import AutoTokenizer

from tester import MetamorphicTesting
from embeddings import PaddlePaddleEmbeddings, HuggingfaceEmbeddings, UformEmbeddings, OptimumEmbeddings, FasttextEmbeddings, CohereEmbeddings, CustomEmbeddings
import jsonlines
from pathlib import Path
import numpy as np
import pandas as pd
import os
from vectorstores import VectorDatabase

embedding_models = ['PaddlePaddle/ernie-3.0-medium-zh',
                    'sgugger/rwkv-430M-pile',
                    'sentence-transformers/all-MiniLM-L6-v2',
                    'unum-cloud/uform-vl-english',
                    'SpanBERT/spanbert-large-cased',
                    'google/electra-large-generator',
                    'sentence-transformers/gtr-t5-large',
                    'sentence-transformers/sentence-t5-large',
                    'sentence-transformers/all-mpnet-base-v2',
                    'tiiuae/falcon-7b',
                    'decapoda-research/llama-7b-hf-4bit']
metamorphic = ['word_swap', 'obj_sub', 'verb_sub', 'nega_exp', 'word_del', 'num_sub', 'err_translate', 'err_nli']
distance_metrics = ['cosine', 'euclidean', 'person', 'manhattan', 'lancewilliams', 'mahalanobis', 'braycurtis']


def load_dataset(path):
    df = pd.read_json(path, lines=True)
    return df[['sentence1', 'sentence2', 'sentence3']].values.tolist()



'''
embedding = PaddlePaddleEmbeddings(model_name="PaddlePaddle/ernie-3.0-medium-zh", cache_folder='models/paddle')
embedding = HuggingfaceEmbeddings(model_name="cross-encoder/quora-distilroberta-base")
embedding = CohereEmbeddings()
embedding = OptimumEmbeddings(model_name='GPTCache/paraphrase-albert-onnx')
'''




if __name__ == "__main__":
    vector_db = 'Annoy'
    distance_metric = None
    # mt = MetamorphicTesting(embedding, distance_metric, vector_db)
    all_dataset = []
    for me in metamorphic:
        dataset = load_dataset('data/MeTMaP/dataset/normal/'+me+'.jsonl')
        all_dataset.append(dataset)
    embedding = CustomEmbeddings(model_name="Cohere_embed-english-v2.0", subsets=metamorphic)
    candidates = np.concatenate([sublist[-2:] for d in all_dataset for sublist in d])
    vb = VectorDatabase(candidates,
                        embedding, vector_db)
    acc = 0
    for dataset in all_dataset:
        for b, p, n in dataset:
            if vb.simulate_retrieval(b)[0].page_content == p:
                acc += 1
    acc /= len(all_dataset) * 5000
    print(acc)


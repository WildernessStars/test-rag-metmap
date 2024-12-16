from questeval.questeval_metric import QuestEval
from typing import List
import jsonlines
import numpy as np


questeval = QuestEval(no_cuda=False)
metamorphic = ['word_swap', 'obj_sub', 'verb_sub', 'nega_exp', 'word_del', 'num_sub', 'err_translate', 'err_nli']


def load_dataset(path) -> List[List[str]]:
    data = []
    with jsonlines.open(path) as f:
        for line in f:
            data.append([line['sentence1'], line['sentence2'], line['sentence3']])
    return data

batch_size = 32
for me in metamorphic:
    dataset = load_dataset('../drive//MyDrive/Colab Notebooks/data/MeTMaP/dataset/normal/'+me+'.jsonl')
    score = []
    score_pos = []
    score_neg = []
    print(me)
    for i in range(0, len(dataset), batch_size):
        batch_base = [case[0] for case in dataset[i:i+batch_size]]
        batch_pos = [case[1] for case in dataset[i:i+batch_size]]
        batch_neg = [case[2] for case in dataset[i:i+batch_size]]
        score_pos.extend(questeval.corpus_questeval(hypothesis=batch_pos, sources=batch_base)['ex_level_scores'])
        score_neg.extend(questeval.corpus_questeval(hypothesis=batch_neg, sources=batch_base)['ex_level_scores'])
        print(i)
    score = [score_pos, score_neg]
    np.save('../drive//MyDrive/Colab Notebooks/QuestEval_scores' + me + '.npy',
            score)

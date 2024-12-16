import typing as t
import numpy as np


class CustomEmbeddings:
    model_name: str = "Cohere/embed-english-v2.0"
    subsets: t.Optional[str] = None

    def __init__(self, model_name, subsets):
        self.model_name = model_name
        self.subsets = subsets
        import pandas as pd
        self.my_dict = {}
        self.embeds = []
        for i in range(0, len(self.subsets)):
            doc_path = 'data/MeTMaP/dataset/normal/'+self.subsets[i]+'.jsonl'
            df = pd.read_json(doc_path, lines=True)
            documents = df[['sentence1', 'sentence2', 'sentence3']].values.flatten().tolist()
            self.my_dict.update({documents[ind]: f'{i}_{ind}' for ind in range(len(documents))})
            embedding_path = '' + self.model_name.replace('/', '_') + '_' + self.subsets[i] + '.npy'
            embeds = np.load(embedding_path)
            self.embeds.append(embeds.reshape(15000, -1))

    def _get_embedding(
            self, text: str
    ) -> t.List[float]:
        doc_ind, text_ind = self.my_dict[text].split("_")
        print(doc_ind, text_ind)
        embeddings = self.embeds[int(doc_ind)][int(text_ind)]

        return embeddings.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._get_embedding(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._get_embedding(text)
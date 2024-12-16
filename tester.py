import time

from embeddings import HuggingfaceEmbeddings
from metrics.distance_metric import DistanceMetricFactory
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import numpy as np

DISTANCE_METRICS = ['cosine', 'euclidean', 'person', 'manhattan', 'lancewilliams', 'braycurtis', 'mahalanobis']


class MetamorphicTesting:
    def __init__(self, embedding: str | Any, metric: str | Any, vector_database=None):
        self.vector_db = vector_database  # Instance of VectorDatabase
        if isinstance(metric, str):
            metric = DistanceMetricFactory.create(metric)
        if isinstance(embedding, str):
            embedding = HuggingfaceEmbeddings(model_name=embedding)
        self.embedding = embedding
        self.embedding_arrays = []
        self.metric = metric  # DistanceMetric instance
        self.cov_matrix = None

    def run_embeddings(self, test_triplets, **kwargs):
        results = []
        cnt = 1
        self.embedding_arrays = []
        if 'embed_document_only' not in kwargs or kwargs['embed_document_only'] is False:
            for base, positive, negative in test_triplets:
                base_vector = self.embedding.embed_query(base)
                positive_vector = self.embedding.embed_query(positive)
                negative_vector = self.embedding.embed_query(negative)
                self.embedding_arrays.append(base_vector)
                self.embedding_arrays.append(positive_vector)
                self.embedding_arrays.append(negative_vector)
                cnt += 1
                print(cnt)
        else:
            flattened_test_triplets = [item for sublist in test_triplets for item in sublist]
            flattened_vectors = []
            for i in range(0, len(flattened_test_triplets), 1000):
                flattened_vectors.extend(
                    self.embedding.embed_documents(
                        flattened_test_triplets[i: min(i+1000, len(flattened_test_triplets))]))
                print(1)
                time.sleep(60)
            self.embedding_arrays = [flattened_vectors[i: i + 3] for i in range(0, len(flattened_vectors), 3)]
        if "subset" in kwargs:
            np.save(self.embedding.model_name.replace('/', '_') + '_' + kwargs['subset'] + '.npy',
                    self.embedding_arrays)
        self.calculate_cov_matrix()
        for i in range(0, len(test_triplets)):
            if "chosen_metric" in kwargs:
                # verify_result
                result = self.verify_result(self.embedding_arrays[3*i],
                                            self.embedding_arrays[3*i + 1], self.embedding_arrays[3*i + 2], **kwargs)
                results.append(result)
            else:
                # parallel_verify
                results_by_metric = self.parallel_verify(self.embedding_arrays[3*i],
                                                         self.embedding_arrays[3*i + 1], self.embedding_arrays[3*i + 2])
                results.append(results_by_metric)
        return results

    def run_predictions(self,  test_triplets, **kwargs):
        self.predictions = []
        score_pos = []
        score_neg = []
        cnt = 0
        for base, positive, negative in test_triplets:
            scores = self.embedding.predict([[base, positive], [base, negative]])
            score_pos.append(scores[0])
            score_neg.append(scores[1])
            cnt += 1
            print(cnt)
        self.predictions.append(score_pos)
        self.predictions.append(score_neg)
        np.save(self.embedding.model_name.replace('/', '_') + '_' + kwargs['subset'] + '.npy',
                self.embedding_arrays)

    def calculate_cov_matrix(self):
        if len(self.embedding_arrays) < 2:
            raise ValueError("Not enough data to calculate covariance matrix")
        self.cov_matrix = np.linalg.inv(np.cov(self.embedding_arrays, rowvar=False))

    def simulate_retrieval(self, test_triplets):
        # todo
        matches = []
        for base, positive, negative in test_triplets:
            match = self.vector_db.simulate_retrieval(base)
            matches.append(match)
        return matches

    def generate_triplets(self, dataset, metamorphic_relation):
        # Use the `TestTripletGenerator` to create (base, positive, negative) triples
        pass

    def verify_result(self, base_vector, positive_vector, negative_vector, **kwargs):
        # Check if match points to positive or negative vector
        if "chosen_metric" in kwargs:
            metric = kwargs.get("chosen_metric")
        else:
            metric = self.metric
        if metric == 'mahalanobis':
            metric = DistanceMetricFactory.create(metric)
            pos_distance = metric(base_vector, positive_vector, self.cov_matrix)
            neg_distance = metric(base_vector, negative_vector, self.cov_matrix)
        else:
            metric = DistanceMetricFactory.create(metric)
            pos_distance = metric(base_vector, positive_vector)
            neg_distance = metric(base_vector, negative_vector)
        if pos_distance < neg_distance:
            return 1, pos_distance, neg_distance
        return 0, pos_distance, neg_distance

    def parallel_verify(self, base_vector, positive_vector, negative_vector):
        results_by_metric = [[] for _ in DISTANCE_METRICS]

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self.verify_result, base_vector, positive_vector, negative_vector, chosen_metric=dm): i
                for i, dm in enumerate(DISTANCE_METRICS)
            }
            for future in futures:
                metric_index = futures[future]
                results_by_metric[metric_index].append(future.result())

        return results_by_metric

    def evaluate_distance(self, result):
        result = np.array(result)
        if len(result) < 10:
            col_means = [np.mean(res, axis=0) for res in result]
        else:
            col_means = np.mean(result, axis=0)
        return col_means

    def evaluate_db(self, all_dataset, vector_db):
        acc = 0
        acc_all = []
        for dataset in all_dataset:
            for b, p, n in dataset:
                candidates = vector_db.simulate_retrieval(b)
                for c in candidates:
                    if c.page_content == p:
                        acc += 1
                        break
                    elif c.page_content == n:
                        break

        acc /= len(all_dataset) * 5000
        acc_all.append(acc)
        return acc_all

    def report(self, result):
        pass

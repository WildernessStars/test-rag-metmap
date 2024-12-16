from scipy.spatial.distance import cosine, euclidean, cityblock, braycurtis
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv


class DistanceMetricFactory:
    @staticmethod
    def create(metric_name):
        if metric_name == 'cosine':
            return lambda vec1, vec2: cosine(vec1, vec2)
        elif metric_name == 'euclidean':
            return lambda vec1, vec2: euclidean(vec1, vec2)
        elif metric_name == 'person':
            return lambda vec1, vec2: 1 - pearsonr(vec1, vec2).statistic
        elif metric_name == 'manhattan':
            return lambda vec1, vec2: cityblock(vec1, vec2)
        elif metric_name == 'lancewilliams':
            return lambda vec1, vec2: DistanceMetricFactory.CanberraDistance(vec1, vec2)
        elif metric_name == 'mahalanobis':
            return lambda vec1, vec2, iv: mahalanobis(vec1, vec2, iv)
        elif metric_name == 'braycurtis':
            return lambda vec1, vec2: braycurtis(vec1, vec2)

    @staticmethod
    def CanberraDistance(x, y):
        x = np.array(x)
        y = np.array(y)
        n = len(x)
        d = 0
        for i in range(len(x)):
            if x[i] == 0 and y[i] == 0:
                d += 0
            else:
                d += abs(x[i] - y[i]) / (abs(x[i]) + abs(y[i]))
        return d


from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from joblib.parallel import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import threading
import numpy as np
from tqdm import tqdm

from clustering.visualization import get_elbow_method


def get_eps(embs, k):
    matrix = cosine_similarity(embs, embs)
    nbrs = NearestNeighbors(n_neighbors=k).fit(matrix)
    distances, indices = nbrs.kneighbors(matrix)
    k_distances = distances[:, -1]
    k_distances.sort()

    plt.plot(list(range(1, len(matrix) + 1)), k_distances)
    plt.xlabel('Values of k')
    plt.ylabel('Distortion')
    plt.savefig('/Users/anastasiaspileva/quality-assessment/clustering/visualize_result/dbscan/graphic_of_eps.png')


class ModelDBSCAN(object):

    def __init__(self, embs, matrix_similar):
        self.embs = embs
        self.matrix_similar = matrix_similar
        self.best_model = DBSCAN()
        self.best_score = 0

    def dbscan_result(self, distance_metric):
        eps_list = np.arange(start=3, stop=3.5, step=0.01)
        min_sample_list = np.arange(start=10, stop=20, step=2)
        max_sil_score = -1
        best_model = DBSCAN()
        for eps_trial in eps_list:
            for min_sample_trial in min_sample_list:
                db = DBSCAN(eps=eps_trial, min_samples=min_sample_trial, metric=distance_metric)
                result = db.fit_predict(self.matrix_similar)
                try:
                    sil_score = silhouette_score(self.matrix_similar, result)
                except ValueError:
                    sil_score = 0
                if sil_score > max_sil_score:
                    max_sil_score = sil_score
                    self.best_model = db

        return best_model, max_sil_score


def run_dbscan(embs, similarity):
    print("run dbscan")
    my_metrics = ['euclidean', 'l2', 'canberra', 'hamming']
    max_sil_score = -1
    modelDBSCAN = ModelDBSCAN(embs, similarity)
    for i in my_metrics:
        model, sil_score = modelDBSCAN.dbscan_result(i)
        if max_sil_score < sil_score:
            max_sil_score = sil_score
            modelDBSCAN.best_model = model

    modelDBSCAN.best_score = max_sil_score
    return modelDBSCAN.best_model.labels_, modelDBSCAN.best_score


class ModelKMEANS(object):

    def __init__(self, embs, matrix_similar):
        self.embs = embs
        self.matrix_similar = matrix_similar
        self.best_model = KMeans()
        self.max_sil_score = -1
        self.count_clusters_list = []
        self.distortions = []

    def run_one_kmeans(self, count_clusters):
        kmeans = KMeans(n_clusters=count_clusters)
        result = kmeans.fit_predict(self.matrix_similar)
        sil_score = silhouette_score(self.matrix_similar, result)

        self.distortions.append(kmeans.inertia_)
        if self.max_sil_score < sil_score:
            self.max_sil_score = sil_score
            self.best_model = kmeans

    def kmeans_result(self):
        count_clusters_list = [2, 3]
        for n_clusters_value in count_clusters_list:
            self.run_one_kmeans(n_clusters_value)

        # get_elbow_method(self.distortions, count_clusters_list)
        return self.best_model.labels_,


def run_kmeans(embs, similarity):
    model_kmeans = ModelKMEANS(embs, similarity)
    return model_kmeans.kmeans_result(), model_kmeans.max_sil_score, model_kmeans.distortions, model_kmeans.count_clusters_list

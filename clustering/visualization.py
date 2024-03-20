from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image
import os
from tqdm import tqdm
#
#
# def visualize(labels, matrix_similar, path):
#     plt.clf()
#     plt.scatter(matrix_similar[:, 0], matrix_similar[:, 1], c=labels, cmap='viridis')
#     plt.title('Distribution')
#     plt.savefig(path)
#
#
# def pca_visualization(X, y, path):
#     # и там, и там нужно
#     plt.clf()
#     pca = PCA(n_components=2)
#     X_reduced = pca.fit_transform(X)
#     plt.figure(figsize=(14, 12))
#     plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y,
#                 edgecolor='none', alpha=0.7, s=40,
#                 cmap=plt.cm.get_cmap('nipy_spectral', 10))
#     plt.colorbar()
#     plt.savefig(path)
#
#
# def visualize_dependency(arr1, arr2, path):
#     # нужно пока только в kmeans
#     # plt.clf()
#     plt.plot(arr1, arr2)
#     plt.title('Dependency')
#     plt.savefig(path)
#
#
# def visualize_distribution(list_width, list_height, path):
#     # Создание графика
#     plt.scatter(list_width, list_height)
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.title('График распределения высоты и ширины изображения')
#     plt.savefig(path)
#
#
def get_elbow_method(distortions, n_clusters):
    plt.figure(figsize=(16, 8))
    plt.plot(n_clusters, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('Метод локтя для определения оптимального количества кластеров')
    # plt.show()
    plt.savefig('/Users/anastasiaspileva/quality-assessment/clustering/visualize_result/kmeans/elbow_method.png')

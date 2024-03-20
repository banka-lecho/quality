import numpy as np
import os
import zipfile
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from clustering.blipModel import run_blip
from clustering.clearing import drop_duplicates, drop_unsuitable_pics
from clustering.clipModel import run_clip
from clustering.clustering_models import run_dbscan, run_kmeans
from clustering.save_read import save_data, read_files
from clustering.stata import normalize_images
from prefect import task, flow
import configparser

# from clustering.visualization import visualize, pca_visualization

warnings.filterwarnings('ignore')


@task
def get_clip_results(path, path_bad_pics):
    """получила эмбеддинги картинок"""
    dict_embs = run_clip(path)
    dict_bad_embs = run_clip(path_bad_pics)
    return dict_embs, dict_bad_embs


@task
def get_blip_results(paths_of_pics):
    """получила описания к картинкам"""
    blip_model = run_blip(paths_of_pics)
    texts = blip_model.texts
    return texts


@task
def clear_data(dict_embs, dict_bad_embs):
    """чистим данные от мусора"""
    dict_embs = drop_duplicates(dict_embs)
    dict_embs = drop_unsuitable_pics(dict_embs, dict_bad_embs)
    return dict_embs


@task
def dbscan(arr_embs, similarity, path_result):
    """" Получаем результаты DBSCAN """

    dbscan_labels, dbscan_sil_score = run_dbscan(arr_embs, similarity)
    # visualize(dbscan_labels, similarity, path_result + 'dbscan/distribution.png')
    # pca_visualization(similarity, dbscan_labels, path_result + 'dbscan/pca.png')
    np.savetxt(path_result + 'dbscan/labels.txt', dbscan_labels)
    return dbscan_sil_score, dbscan_labels


@task
def kmeans(arr_embs, similarity):
    """" Получаем результаты Kmeans"""

    kmeans_labels, kmeans_sil_score, distortions, count_clusters = run_kmeans(arr_embs, similarity)
    # visualize(kmeans_labels, similarity, path_result + 'kmeans/distribution.png')
    # pca_visualization(similarity, kmeans_labels, path_result + 'kmeans/pca.png')
    return kmeans_labels, distortions, count_clusters


def zip_folder(folder_path, output_name):
    """" Сохраняем результат: лейблы и эмбеддинги"""

    with zipfile.ZipFile(output_name, 'w') as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".npz"):
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, folder_path))


def check_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'Директория {path} создана')


# def run_clustering(arr_embs, similarity, path_result):
#     # dbscan_score, dbscan_labels = dbscan(arr_embs, similarity)
#     kmeans_score, kmeans_labels = kmeans(arr_embs, similarity, path_result)
#     # final_labels = dbscan_labels if (dbscan_score > kmeans_score) else kmeans_labels
#     return kmeans_labels

@flow()
def start(path, path_bad_pics, path_statistics, path_result):
    # чекаем распределение картинок
    # list_error = get_distribution(path, path_statistics + 'dist_first')
    list_width, list_height = normalize_images(path, path_statistics + 'dist_draft')
    save_data(list_width, path_result + 'dimensions_pictures/list_width.pkl')
    save_data(list_height, path_result + 'dimensions_pictures/list_height.pkl')

    # получаем эмбеддинги
    os.listdir(path)

    dict_embs, dict_bad_embs = get_clip_results(path, path_bad_pics)

    save_data(dict_embs, path_result + 'embs/embs.pkl')
    save_data(dict_bad_embs, path_result + 'embs/bad_embs.pkl')

    dict_embs = read_files(path_result + 'embs/embs.pkl')
    dict_bad_embs = read_files(path_result + 'embs/bad_embs.pkl')

    # чистим эмбеддинги картинок
    dict_embs = clear_data(dict_embs, dict_bad_embs)
    save_data(dict_embs, path_result + 'embs/clear_embs.pkl')

    clear_image_embs = read_files(path_result + 'embs/clear_embs.pkl')

    # заспукаем кластеризацию
    keys = sorted(clear_image_embs.keys())
    embeddings = [clear_image_embs[key].flatten() for key in keys]

    similarity = cosine_similarity(embeddings, embeddings)
    labels, distortions, count_clusters = kmeans(embeddings, similarity)
    save_data(labels, path_result + 'kmeans/labels.txt')
    save_data(distortions, path_result + 'kmeans/distortions.pkl')
    save_data(count_clusters, path_result + 'kmeans/count_clusters.pkl')

    # получаем описания
    run_blip(keys)


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('paths.txt', encoding='utf-8')

    path_dirty_pics = config.get('Paths', 'path_dirty_pics')
    path = config.get('Paths', 'path')
    path_bad_pics = config.get('Paths', 'path_bad_pics')
    path_statistics = config.get('Paths', 'path_statistics')
    path_result = config.get('Paths', 'path_result')
    path_visualization = config.get('Paths', 'path_visualization')

    start(path, path_bad_pics, path_statistics, path_result)

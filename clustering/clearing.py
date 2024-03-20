import shutil
import os
import copy
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm


def drop_duplicates(dict_all_embs):
    print("drop duplicates")
    print(len(dict_all_embs))
    dict_copy = dict_all_embs.copy()
    for key1, emb1 in dict_copy.items():
        for key2, emb2 in dict_copy.items():
            if key1 != key2:
                if key1 in dict_all_embs.keys():
                    if cosine_similarity(emb1, emb2) > 0.99999999999999999:
                        del dict_all_embs[key2]

    return dict_all_embs


def drop_unsuitable_pics(dict_all_embs, dict_bad_embs):
    print("drop unsuitable pics")
    dict_copy = dict_all_embs.copy()
    for key1, emb1 in dict_copy.items():
        for key2, emb2 in dict_bad_embs.items():
            if key1 in dict_all_embs.keys():
                if cosine_similarity(emb1, emb2) > 0.999:
                    del dict_all_embs[key1]

    return dict_all_embs

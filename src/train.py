import os
import json
import argparse
import numpy as np
import gensim
from sklearn.cluster import DBSCAN, KMeans, HDBSCAN
from sklearn.metrics.cluster import homogeneity_score, completeness_score, v_measure_score
import pymorphy2
import gensim
from gensim.test.utils import datapath
from utils import *


def build_method(args, vectors=None, labels=None):
    if args.method == "dbscan":
        if args.search and args.method != "hdbscan":
            (eps, min_samples), _ = dbscan_search(vectors, labels)
        else:
            eps, min_samples = args.eps, args.min_samples
        return DBSCAN(eps=eps, min_samples=min_samples), (eps, min_samples)
        
    elif args.method == "hdbscan":
        return HDBSCAN(min_cluster_size=3, 
                          metric='cosine', 
                          cluster_selection_epsilon=0.4, 
                          min_samples=2,), None
        
    elif args.method == "kmeans":
        if args.search:
            _, cls = dbscan_search(vectors, labels)
        else:
            cls = 5
        return KMeans(n_clusters=cls,
               max_iter=300, 
               tol=1e-4, 
               init='random', 
               n_init=10, 
               random_state=42), None
    else:
        return None


def dbscan_search(vectors, label_list, min_samples_min=3, 
                  min_samples_max=22, eps_min=0.05, eps_max=1.2, eps_step=0.01):
    max_v = -100
    save_v = None
    num_clusters = -1
    for eps in np.arange(eps_min, eps_max, eps_step):
        for min_samples in range(min_samples_min, min_samples_max):
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            y_pred = dbscan.fit_predict(vectors)
            
            if np.unique(y_pred).size < 3:
                continue
                
            curr_m = v_measure_score(label_list, y_pred)
            
            if curr_m > max_v:
                max_v = curr_m
                save_v = (eps, min_samples)
                num_clusters = np.max(y_pred) - np.min(y_pred) + 1

    return save_v, num_clusters

def main():
    parser = argparse.ArgumentParser(description="Cognates clustering")
    parser.add_argument("-m", "--model-path", type=str, help="Path to the fasttext model", default='../213/model.model')
    parser.add_argument( "--model-type", type=str, help="FastText or Word2Vec", default='fasttext')
    parser.add_argument("-d", "--data", type=str, help="train dataset folder", default="./dataset")
    parser.add_argument("-i", '--input-folder', type=str, help="Json with input groups", default='dataset')
    parser.add_argument('--min-samples', type=int, help="Min samples argument for DBSCAN", default=3)
    parser.add_argument('--eps', type=float, help="Eps argument for DBSCAN", default=0.9)
    parser.add_argument('--search', action='store_true', default=False)
    parser.add_argument("--out-folder", type=str, default="./out", help="path to write the resulting clusters")
    parser.add_argument('--method', type=str, default="dbscan", help='method for word group clustering')
    args = parser.parse_args()

    if args.model_type == "fasttext":
        model = gensim.models.fasttext.FastTextKeyedVectors.load(args.model_path)
    else: 
        model = gensim.models.KeyedVectors.load_word2vec_format(datapath(args.model_path), 
                                                            binary=True)

    output = {}
    files = [t for t in os.listdir(os.path.join(args.data, "groups")) if 'txt' in t]
    mean_h, mean_c, mean_v = 0, 0, 0
    eps, min_samples = 0, 0
    
    for f in files:
        word_list = read_words(os.path.join(args.data, "groups", f))
        vectors, words, missed = get_embeddings(model, word_list, 
                                        fasttext=(args.model_type == "fasttext"))
        labels = read_labels(os.path.join(args.data,"labels", f), words, 
                                        fasttext=(args.model_type == "fasttext"))

        method, params = build_method(args, vectors, labels)
        y_pred = method.fit_predict(vectors)
            
        h = homogeneity_score(labels, y_pred)
        c = completeness_score(labels, y_pred)
        v = v_measure_score(labels, y_pred)
        mean_h += h
        mean_c += c
        mean_v += v
        print("Homogeneity: {}, Completeness: {}, v_measure: {}".format(h, c, v))
        
        sorted_list = {i: [] for i in range(np.min(y_pred), np.max(y_pred) + 1)}
        for j, w in enumerate(words):
            sorted_list[int(y_pred[j])].append(w)

        os.makedirs(args.out_folder, exist_ok=True)
        with open(f'./{args.out_folder}/families_{args.model_type}_' + f, "w") as out_file:
            preds = []
            for j, w in enumerate(words):
                preds.append((w, labels[j], y_pred[j]))
            preds.sort(key=lambda x:x[1])
                
            for j, w in enumerate(words):
                out_file.write(w + " " + str(preds[j][1]) + " "  + str(preds[j][2]) + "\n")
                
            for w in missed:
                out_file.write(w + '\n')

        if args.method == "dbscan":
            eps += params[0]
            min_samples += params[1]
            print("Optimal params are: {}, {}".format(params[0], params[1]))
            
        print(sorted_list)

    print("Mean H:{}, C:{}, V:{}".format(mean_h/len(files), mean_c/len(files), mean_v/len(files)))

    if args.method =="dbscan":
        print("Mean eps:{}, min_samples:{}".format(eps/len(files), min_samples/len(files)))
            
if __name__ == "__main__":
    main()
import json
import argparse
import numpy as np
import gensim
from gensim.test.utils import datapath
from sklearn.cluster import DBSCAN
from utils import *
import copy

def get_centroids(model, dataset):
    embeds = {}
    for k in dataset.keys():
        if k == "missed":
            continue
        vec = 0
        n = 0
        for w in dataset[k]:
            try:
                v = model[w]
                vec += v
                n += 1
            except Exception as e:
                continue
        if n > 0:
            embeds[k] = vec/n
    return embeds

def get_similar(embeds, model, word):
    try:
        a = model[word]
    except Exception as e:
        print("No such word")
        return
    sims = []
    for k in embeds.keys():
        b = embeds[k]
        cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
        sims.append((k, cos_sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)
    return sims
    
def main():

    parser = argparse.ArgumentParser(description="Cognates clustering")
    parser.add_argument("-m", "--model-path", type=str, help="Path to the fasttext model", default='../213/model.model')
    parser.add_argument("-i", '--input-file', type=str, help="Json with input groups", default='filtered_groups_20.json')
    parser.add_argument("-o", '--output', type=str, help="path to the json output", default='output_clusters_replenished.json')
    args = parser.parse_args()
    
    model = gensim.models.fasttext.FastTextKeyedVectors.load(args.model_path)
    input_groups = read_json(args)
    
    output = {}
    
    for i in input_groups.keys():
        
        groups = copy.deepcopy(input_groups[i])
        centroids = get_centroids(model, input_groups[i])
        
        for w in input_groups[i]["missed"]:    
            sims = get_similar(centroids, model, w)[0][0]
            groups[sims].append(w)

        groups.pop("missed", None)

        output[i] = groups

    with open(args.output, 'w') as f:
        json.dump(output, f)
            
if __name__ == "__main__":
    main()

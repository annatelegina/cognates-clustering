import json
import argparse
import numpy as np
import gensim
from gensim.test.utils import datapath
from sklearn.cluster import DBSCAN
from utils import *

def main():

    parser = argparse.ArgumentParser(description="Cognates clustering")
    parser.add_argument("-m", "--model-path", type=str, help="Path to the fasttext model", default='../213/model.model')
    parser.add_argument( "--model-type", type=str, help="FastText or Word2Vec", default='word2vec')
    parser.add_argument("-i", '--input-file', type=str, help="Json with input groups", default='filtered_groups_20.json')
    parser.add_argument("-o", '--output', type=str, help="path to the json output", default='output_clusters.json')
    parser.add_argument('--min-samples', type=int, help="Min samples argument for DBSCAN", default=3)
    parser.add_argument('--eps', type=float, help="Eps argument for DBSCAN", default=1.1)
    args = parser.parse_args()
    
    if args.model_type == "fasttext":
        model = gensim.models.fasttext.FastTextKeyedVectors.load(args.model_path)
    else: 
        model = gensim.models.KeyedVectors.load_word2vec_format(datapath(args.model_path), 
                                                            binary=True)
    _, file_ext = os.path.splitext(args.input_file)

    if file_ext == ".txt":
        input_groups = {}
        input_groups[0] = read_words(args.input_file)
    elif file_ext == ".json":
        input_groups = read_json(args)
    
    filtered = {}
    output = {}
    
    for i in input_groups.keys():
        vectors, found_words, missed = get_embeddings(model, input_groups[i], 
                                        fasttext=(args.model_type == "fasttext"))
        
        dbscan = DBSCAN(eps=args.eps, min_samples=args.min_samples)
        y_pred = dbscan.fit_predict(vectors)
        
        sorted_list = {i: [] for i in range(np.min(y_pred), np.max(y_pred) + 1)}
        
        for j, w in enumerate(found_words):
            sorted_list[int(y_pred[j])].append(w)

        sorted_list["missed"] = []

        for w in missed:
            sorted_list["missed"].append(w)
        output[i] = sorted_list
        
    with open(args.output, 'w') as f:
        json.dump(output, f)
            
if __name__ == "__main__":
    main()

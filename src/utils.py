import os
import numpy as np
import pymorphy2

pos_dict = {
    "NOUN": "NOUN", 
    "ADVB": "ADVB", 
    "INFN": "VERB",
    "VERB": "VERB",
    "PRTF": "ADJ", 
    "ADJF": "ADJ", 
    "ADJS": "ADJ", 
    "COMP": "ADJ",
}

morph = pymorphy2.MorphAnalyzer()

def transform(w):
    w = w.split('/')
    word = ''
    for part in w:
        word += part.split(':')[0]
    return word

def read_words(filepath):
    i = 0
    roots = []
    words = []
    with open(filepath, 'r') as f:
        for line in f:
            if i == 0:
                roots = line.strip().split(',')
            else:
                word = transform(line)
                words.append(word)
            i += 1

    return words

def read_labels(filepath, words, fasttext=True):
    global pos_dict, morph
    i = 0
    label_target = {}

    ordered_w = []
    with open(filepath, 'r') as f:
        for line in f:
            if ':' not in line:
                i += 1
                continue
            word = transform(line)
            if not fasttext:
                tag = morph.parse(word)[0].tag.POS
                if tag not in pos_dict.keys():
                    continue
            search_w = word if fasttext else word + "_" + pos_dict[tag]
            if search_w in words:
                label_target[search_w] = i
                ordered_w.append(search_w)

    i += 1
    for w in words:
        if w not in label_target.keys():
            label_target[w] = i
            ordered_w.append(w)
            
    label_list = []
    for w in words:
        label_list.append(label_target[w])
        
    return label_list
    
def get_embeddings(model, words, fasttext=True):
    global pos_dict, morph
    vectors = []
    found_words = []
    missed_words = []
    for ww in words:
        w = transform(ww)
        try:
            if fasttext:
                v = model[w]
                found_words.append(w)
            else:
                tag = morph.parse(w)[0].tag.POS
                if tag not in pos_dict.keys():
                    continue
                v = model.get_vector(w + "_" + pos_dict[tag])
                found_words.append(w + "_" + pos_dict[tag])
            norm = np.linalg.norm(v)
            vectors.append(v/norm)
        except Exception as e:
            missed_words.append(w)
            continue

    return vectors, found_words, missed_words
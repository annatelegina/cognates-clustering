# Recognizing Cognates in Russian

This project is dedicated to the automatic recognition of cognates (words sharing the same root) in Russian.
## Steps

1. **Data preprocessing**  
   - The RuMorphs-Lemmas (≈96 thousand of words) dataset with morpheme segmentation is used.  
   - Words with multiple roots and numerals are excluded, leaving ≈77.8k words.

2. **Forming sets of words with allomorphic and homonymous roots**  
   - All unique roots are extracted.  
   - Merging rules are applied based on known alternations of consonants and vowels in roots.  
   - As a result, disjoint sets of words with potentially common roots. 

3. **Word embeddings clustering**  
   - For each word in every set, vector representations are taken from Word2Vec and FastText distributional semantic models.  
   - Clusters are extracted using **DBSCAN**, **HDBSCAN**, and **K-means**. DBSCAN parameters are optimized, HDBSCAN is used with default settings from scikit-learn, and the number of clusters for K-means is derived from DBSCAN results. 
   - The best results according to the V-measure were obtained with *DBSCAN + Word2Vec-Tayga*.  

4. **Cognate groups replenishment**  
   - For words without embeddings, the FastText model is used.  
   - The target cognate group is determined by cosine similarity between the word vector and cluster centroids. 

## DBSCAN parameters optimization

The parameters `eps` and `min_samples` were tuned via grid search over predefined ranges (`eps`: (0.05, 1.2) with step 0.01, `min_samples`: (3, 22) with step 1).
For each parameter pair, clustering was run on annotated datasets, and the quality was evaluated using **homogeneity**, **completeness**, and **V-measure**. 

The final optimal parameters were chosen as the **average values** across all annotated datasets.

| Model | Word2vec Ruswiki | Word2vec Tayga | FastText Tayga |
| ------ | ---------------- | -------------- | -------------- |
|eps |1.1 | 1.1 | 0.9 |
| min_samples | 3 | 4 | 3 |


## Usage

1. **Download distributional semantic models**


```bash
cd models && ./download_models.sh
```

2. **Parameters search and experiments for annotated data**  

```bash
python src/train.py --search --model-type [word2vec|fasttext] --model-path PATH_TO_MODEL --method [dbscan|hdbscan|kmeans]
``` 
3. **Clustering word sets with DBSCAN**  

```bash
python src/cluster.py -i INPUT_FILE_PATH --model-type [word2vec|fasttext] --model-path PATH_TO_MODEL --eps EPS --min-samples MIN_SAMPLES
```

4. **Cognate groups replenishment with FastText model**  

```bash
python src/replenish.py -i INPUT_FILE_PATH --model-path PATH_TO_FASTTEXT_MODEL -o PATH_TO_OUTPUT_FILE
``` 

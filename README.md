# hyperbolic-learning
-----

**It has been recently established that many real-world networks have a latent geometric structure that resembles negatively curved hyperbolic spaces. Therefore, complex networks, and particularly the hierarchical relationships often found within, can often be more accurately represented by embedding graphs in hyperbolic geometry, rather than flat Euclidean space.**

**The goal of this project is to provide Python implementations for a few recently published algorithms that leverage hyperbolic geometry for machine learning and network analysis. Several examples are given with real-world datasets, however; the time complexity is far from optimized and this repository is primarily for research purposes - specifically investigating how to integrate downstream supervised learning methods with hyperbolic embeddings.**

![IllinformedHalfAnemone-size_restricted](https://user-images.githubusercontent.com/16658498/67390992-564d8880-f563-11e9-95af-a959085e72c1.gif)

# Contents

## Models
* __Poincaré Embeddings__: 
  - Mostly an exploration of the hyperbolic embedding approach used in __[1]__.
  - Available implementation in the `gensim` library and a PyTorch version released by the authors [here](https://github.com/facebookresearch/poincare-embeddings).
  
* __Hyperbolic Multidimensional Scaling__:
  - Solves for embedding coordinates in Poincaré disk with hyperbolic distances that preserve input dissimilarities __[2]__.
  
* __K-Means Clustering in the Hyperboloid Model__:
  - Optimization approach using Frechet means to define a centroid/center of mass in hyperbolic space __[3, 4]__.
  
* __Hyperbolic Support Vector Machine__ - 
  - Linear hyperbolic SVC based on formulating the max-margin optimization problem in hyperbolic geometry __[5]__.
  - Uses projected gradient descent to define optimal decision boundary to perform supervised learning in hyperbolic space.
  
* __Embedding Graphs in Lorentzian Spacetime__ -  
  - An algorithm based on notions of causality in the Minkowski spacetime formulation of special relativity __[6]__.
  - Used to embed directed acyclic graphs where nodes are represented by space-like and time-like coordinates. 

![hep-th_citation_network](https://user-images.githubusercontent.com/16658498/65956193-6fa16000-e40f-11e9-935b-a518a77b6525.png)

![poincare_kmeans](https://user-images.githubusercontent.com/16658498/62563652-11aa2f00-b849-11e9-93e5-4665f9020052.png)

## Datasets
- Zachary Karate Club Network
- WordNet
- Enron Email Corpus
- Polbooks Network
- arXiv Citation Network
- Synthetic generated data (sklearn.make_datasets, networkx.generators, etc.)

## Dependencies
- Models are designed based on the sklearn estimator API (`sklearn ` generally used only in rare, non-essential cases)
- `Networkx` is used to generate & display graphs

## References

__[1]__ Nickel, Kiela. "Poincaré embeddings for learning hierarchical representations" (2017). [arXiv](https://arxiv.org/pdf/1705.08039.pdf).

__[2]__ A. Cvetkovski and M. Crovella. Multidimensional scaling in the Poincaré disk. arXiv:1105.5332, 2011.

__[3]__ "Learning graph-structured data using Poincaré embeddings and Riemannian K-means algorithms". Hatem Hajri, Hadi Zaatiti, Georges Hebrail (2019) [arXiv](https://arxiv.org/abs/1907.01662).

__[4]__ Wilson, Benjamin R. and Matthias Leimeister. “Gradient descent in hyperbolic space.” (2018).

__[5]__ "Large-margin classification in hyperbolic space". Cho, H., Demeo, B., Peng, J., Berger, B. CoRR abs/1806.00437 (2018).

__[6]__ Clough JR, Evans TS (2017) Embedding graphs in Lorentzian spacetime. PLoS ONE 12(11):e0187301. https://doi.org/10.1371/journal.pone.0187301.

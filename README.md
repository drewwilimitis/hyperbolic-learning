# hyperbolic-learning

**It has been recently established that many real-world networks have a latent geometric structure that resembles negatively curved hyperbolic spaces. Therefore, complex networks, and particularly the hierarchical relationships often found within, can often be more accurately represented by embedding graphs in hyperbolic geometry, rather than flat Euclidean space.**

**The goal of this project is to provide Python implementations for a few recently published algorithms that leverage hyperbolic geometry for machine learning and network analysis. Several examples are given with real-world datasets, however; the time complexity is far from optimized and this repository is primarily for research purposes - specifically investigating how to integrate downstream supervised learning methods with hyperbolic embeddings.**

![IllinformedHalfAnemone-size_restricted](https://user-images.githubusercontent.com/16658498/67390992-564d8880-f563-11e9-95af-a959085e72c1.gif)

## Contents
------

### Models:
* __Poincaré Embeddings__ - Implementing hyperbolic embeddings outlined by Nickel and Kiela from Facebook AI Research in "Poincaré embeddings for learning hierarchical representations" (2017)  https://arxiv.org/pdf/1705.08039.pdf. Mostly exploratory rather than operational, especially given an available implementation in the `gensim` library and a PyTorch implementation released by the authors at https://github.com/facebookresearch/poincare-embeddings.

* __Hyperbolic Multidimensional Scaling__ - My implementation of the Poincaré disk MDS algorithm discussed by A. Cvetkovski and M. Crovella in "Multidimensional scaling in the Poincaré disk." arXiv:1105.5332, 2011

* __K-Means Clustering in the Hyperboloid Model__ - A modified version of the Hyperbolic KMeans algorithm recently presented in "Learning graph-structured data using Poincaré embeddings and Riemannian K-means algorithms". Hatem Hajri, Hadi Zaatiti, Georges Hebrail (2019) https://arxiv.org/abs/1907.01662. Primarily following the optimization and Frechet mean approach outlined by Wilson, Benjamin R. and Matthias Leimeister. “Gradient descent in hyperbolic space.” (2018).

* __Hyperbolic Support Vector Machine__ - Python implementation of "Large-margin classification in hyperbolic space". Cho, H., Demeo, B., Peng, J., Berger, B. CoRR abs/1806.00437 (2018).

* __Embedding Graphs in Lorentzian Spacetime__ - Implementing the DAG spacetime embedding algorithm published by Clough JR, Evans TS (2017) Embedding graphs in Lorentzian spacetime. PLoS ONE 12(11):e0187301. https://doi.org/10.1371/journal.pone.0187301.

![hep-th_citation_network](https://user-images.githubusercontent.com/16658498/65956193-6fa16000-e40f-11e9-935b-a518a77b6525.png)

![poincare_kmeans](https://user-images.githubusercontent.com/16658498/62563652-11aa2f00-b849-11e9-93e5-4665f9020052.png)

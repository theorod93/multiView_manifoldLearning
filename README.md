# Multi-view Data Visualisation via Manifold Learning 
Contains the code of "Multi-view Data Visualisation via Manifold Learning" by T. Rodosthenous, V. Shahrezaei, M. Evangelou. arXiv e-prints,arXiv:2101.06763, 2021.

## Abstract
Non-linear dimensionality reduction can be performed by _manifold learning_ approaches, such as Stochastic Neighbour Embedding (SNE), Locally Linear Embedding (LLE), and Isometric Feature Mapping (ISOMAP). These methods aim to produce two or three latent embeddings, primarily to visualise the data in intelligible representations. This manuscript proposes extensions of Student's t-distributed SNE (t-SNE), LLE, and ISOMAP, for dimensionality reduction and visualisation of _multi-view_ data. Multi-view data refers to multiple types of data generated from the same samples.
  
The proposed multi-view approaches provide more comprehensible projections of the samples compared to the ones obtained by visualising each data-view separately. Commonly,  visualisation is used for identifying underlying patterns within the samples. By incorporating the obtained low-dimensional embeddings from the multi-view manifold approaches into the $K$-means clustering algorithm, it is shown that clusters of the samples are accurately identified. Through extensive comparisons of novel and existing multi-view manifold learning algorithms on real and synthetic data, the proposed multi-view extension of t-SNE, named _multi-SNE_, is found to have the best performance, quantified both qualitatively and quantitatively by assessing the clusterings obtained. 

The applicability of multi-SNE is illustrated by its implementation in the newly developed and challenging multi-omics single-cell data. The aim is to visualise and identify cell heterogeneity and cell types in biological tissues relevant to health and disease. In this application, multi-SNE provides an improved performance over single-view manifold learning approaches and a promising solution for unified clustering of multi-omics single-cell data.

## Functions
This directory contains the functions for the single-view and multi-view manifold learning algorithms that have been used in the aforementioned manuscript. The functions are separated by the original single-view solutions from which their multi-view extensions have been based.

- **SNE.py**: SNE-based
  - t-SNE
  - m-SNE
  - multi-SNE
- **LLE.py**: LLE-based
  - LLE
  - m-LLE
  - multi-LLE
- **ISOMAP.py**: ISOMAP-based
  - ISOMAP
  - m-ISOMAP
  - multi-ISOMAP

## Simulations
This directory provides an R code that was used in simulating data under various conditions/scenarios.

## Examples
For all the real and simulated data, the same approach was taken. This directory provides examples of applying all multi-view manifold learning algorithms. Although the examples presented are based on exploring the Caltech7 dataset, the same methodology was used for all datasets presented in the manuscript. Please note that for simplicity a single value for the corresponding tuning parameter was used in each algorithm (perplexity for SNE-based solutions, and the number of neighbours for ISOMAP-based and LLE-based solutions). However, practically a range of values should be tested as is described in the manuscript. 

## Real Data
- Handwritten Digits: https://archive.ics.uci.edu/ml/datasets/Multiple+Features
- Caltech7: https://github.com/yeqinglee/mvdata
- Cancer Types: http://compbio.cs.toronto.edu/SNF/SNF/Software.html


# HierarchyOfSciences

This repository contains all code used to produce the results in my 2021 BSc Artificial Intelligence thesis "Lexical Semantic Difference in the Hierarchy of Sciences". (File available upon request).

The three corpora can be compiled with `Data.ipynb` which produces the files `jstor.txt`, `arxiv.txt` and `reuters.txt` containing political science abstracts, physics abstracts and newspaper articles respectively.

`DataAnalysis.ipynb` can then be used to perform a frequency based analysis upon these three files.

The local neighbourhood similarity measures Cosine Similarity (CS) and Nearest Neighbours (NN) are computed with the following files: `run.sh`, `word2vec.py`, `CS.py` and `NN.py`. These files are modified code from [Gonen et al.](https://www.aclweb.org/anthology/2020.acl-main.51/)

To perform all similarity measures (NN vs CS and Political Science vs Physics) execute the following commands:

`bash run.sh train`
`bash run.sh detect NN arxiv`
`bash run.sh detect NN jstor`
`bash run.sh detect CS arxiv`
`bash run.sh detect CS jstor`

Lastly, the resulting lists from the commands above can be plotted and analysed with `SimilarityAnalysis.ipynb'.

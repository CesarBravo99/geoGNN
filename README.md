# Geometric Graph Neural Networks

Thesis project repository

## Pending Meeting Tasks (TO DO)

- Revisar algoritmos de pooling de grafos
- Añadir preprocesamiento de los datasets para entrenar GNNS
- Estarandarizar los inputs de las diferentes arquitecturas de GNN para el caso de PROTEINS dataset
- Implementar k-disGNN y geoGNN

## Work Log
***
- 2-jan:
    - Read [5] for graph pooling
    - Adapt graphSAGE for PROTEINS dataset
    - Refactor of GAT

- 1-jan:
    - Review and refactor of graphSAGE [4]. Add MessagePassing class

- 28-dic:
    - Review `Mojo🔥` programming languange documentation for possible High efficient GNN implementation.
      
- 26-dic: 
    - Added QM9.ipynb
    - Refactor of datasets scripts by integrate `PyTorch Geometric` lib. for dataset managing.
    - Added MD17.ipynb, this .ipynb describes the MD17 benzene dataset
    - Added DD.ipnb, this .ipynb describes the DD dataset
    - Connect Github repo with Paperspace workspace

- 25-dic:
    - Added cora.ipynb, this .ipynb describes the cora dataset
***


## Bibliography

1. Rose, V. D., Kozachinskiy, A., Rojas, C., Petrache, M., & Barceló, P. (2023). Three iterations of $(d-1)$-WL test distinguish non isometric clouds of $d$-dimensional points. [arXiv](https://doi.org/10.48550/ARXIV.2303.12853)

2. Li, Z., Wang, X., Huang, Y., & Zhang, M. (2023). Is Distance Matrix Enough for Geometric Deep Learning?. [arXiv](https://doi.org/10.48550/ARXIV.2302.05743)

3. Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2018). Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. [arXiv](https://doi.org/10.48550/ARXIV.1810.02244)

4. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. [arXiv](https://doi.org/10.48550/ARXIV.1706.02216)

5. Liu, C., Zhan, Y., Wu, J., Li, C., Du, B., Hu, W., Liu, T., & Tao, D. (2022). Graph Pooling for Graph Neural Networks: Progress, Challenges, and Opportunities. [arXiv](https://doi.org/10.48550/ARXIV.2204.07321)

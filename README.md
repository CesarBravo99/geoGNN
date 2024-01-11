# Geometric Graph Neural Networks

Thesis project repository

## Pending Meeting Tasks (TO DO)

- Añadir reset de parámetros
- Hacer experimento de entrenamiento con y sin batches normalizando por el nro de FLOPS
- Revisar qué hace la función de plotting
- Investigar sobre edge prediction
- Investigar sobre qué hace la función de pérdida de pytorch
- Realizar experimento para medir sensibilidad de poolings sobre grafos construidos a mano
- Implementar `k-disGNN` [2] y `geoGNN` [1]

## Work Log
***
- 12-jan
  - 👥 Meeting
    
- 11-jan
   - 📚 Read [] for better undestanding of edge prediction

- 9-jan:
   - 👥 Meeting
   - 📚 Code review of [2]
   - ✨ Added graph_prediciont_regresion.ipynb
   - 📚 Read `PyTorch Geometric` documentation for transform and filter PyG datasets

- 8-jan:
   - ✨ Added node_classification.ipynb, graph_prediciont_binary.ipynb and basicGNN.py
   - 🔥 Remove old code
   - ⚡ Permanently install Python libraries on Paperspace
  
- 7-jan:
   -  📚 Read `PyTorch Geometric` documentation on datasets, layars, models and run notebooks examples on Paperspace using graphcore IPU 
   -  🔥 Deprecate data folder

- 4-jan:
   - 👥 Meeting
     
- 3-jan:
    - 📚 Read [9, 10, 11, 12, 13] for graph pooling and readout
    
- 2-jan:
    - 📚 Read [5, 6, 7, 8] for graph pooling and readout
    - ✨ Adapt `graphSAGE` for `PROTEINS` dataset
    - ♻️ Refactor of `GAT`. Work in progress
- 1-jan:
    - ♻️ Review and refactor of `graphSAGE` [4]. Add MessagePassing class

- 28-dic:
    - Review `Mojo🔥` programming languange documentation for possible High efficient GNN implementation.
      
- 26-dic: 
    - ✨ Added `QM9.ipynb`
    - ♻️ Refactor of datasets scripts by integrate `PyTorch Geometric` lib. for dataset managing.
    - ✨ Added `MD17.ipynb`, this .ipynb describes the MD17 benzene dataset
    - ✨ Added `DD.ipnb`, this .ipynb describes the DD dataset
    - ✨ Connect _Github_ repo with _Paperspace_ workspace

- 25-dic:
    - ✨ Added `Cora.ipynb`, this .ipynb describes the cora dataset
***


## Bibliography

1. Rose, V. D., Kozachinskiy, A., Rojas, C., Petrache, M., & Barceló, P. (2023). Three iterations of $(d-1)$-WL test distinguish non isometric clouds of $d$-dimensional points. [arXiv](https://doi.org/10.48550/ARXIV.2303.12853)

2. Li, Z., Wang, X., Huang, Y., & Zhang, M. (2023). Is Distance Matrix Enough for Geometric Deep Learning?. [arXiv](https://doi.org/10.48550/ARXIV.2302.05743)

3. Morris, C., Ritzert, M., Fey, M., Hamilton, W. L., Lenssen, J. E., Rattan, G., & Grohe, M. (2018). Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks. [arXiv](https://doi.org/10.48550/ARXIV.1810.02244)

4. Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. [arXiv](https://doi.org/10.48550/ARXIV.1706.02216)

5. Liu, C., Zhan, Y., Wu, J., Li, C., Du, B., Hu, W., Liu, T., & Tao, D. (2022). Graph Pooling for Graph Neural Networks: Progress, Challenges, and Opportunities. [arXiv](https://doi.org/10.48550/ARXIV.2204.07321)

6. Grattarola, D., Zambon, D., Bianchi, F. M., & Alippi, C. (2021). Understanding Pooling in Graph Neural Networks. [arXiv](https://doi.org/10.48550/ARXIV.2110.05292)

7. Ying, R., You, J., Morris, C., Ren, X., Hamilton, W. L., & Leskovec, J. (2018). Hierarchical Graph Representation Learning with Differentiable Pooling [arXiv](https://doi.org/10.48550/ARXIV.1806.08804)

8.  Buterez, D., Janet, J. P., Kiddle, S. J., Oglic, D., & Liò, P. (2022). Graph Neural Networks with Adaptive Readouts. [arXiv](https://doi.org/10.48550/ARXIV.2211.04952)

9.  Ju, W., Fang, Z., Gu, Y., Liu, Z., Long, Q., Qiao, Z., Qin, Y., Shen, J., Sun, F., Xiao, Z., Yang, J., Yuan, J., Zhao, Y., Luo, X., & Zhang, M. (2023). A Comprehensive Survey on Deep Graph Representation Learning (Version 2). [arXiv](https://doi.org/10.48550/ARXIV.2304.05055)

10. Hamilton, William L. (2023). Graph Representation Learning. Synthesis Lectures on Artificial Intelligence and Machine Learning. [link](https://www.cs.mcgill.ca/~wlh/grl_book/)

11. Mesquita, D., Souza, A. H., & Kaski, S. (2020). Rethinking pooling in graph neural networks (Version 1). [arXiv](https://doi.org/10.48550/ARXIV.2010.11418)

12. Pal, S., Malekmohammadi, S., Regol, F., Zhang, Y., Xu, Y., & Coates, M. (2020). Non-Parametric Graph Learning for Bayesian Graph Neural Networks. [arXiv](https://doi.org/10.48550/ARXIV.2006.13335)

13. Zhang, Z., Bu, J., Ester, M., Zhang, J., Yao, C., Yu, Z., & Wang, C. (2019). Hierarchical Graph Pooling with Structure Learning. [arXiv](https://doi.org/10.48550/ARXIV.1911.05954)

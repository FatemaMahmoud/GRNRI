# GRNRI
## Abstract
The reconstruction of gene regulatory networks (GRNs) from gene expression data is a challenging problem in systems biology. Here we propose GRNRI: an unsupervised model that learns to infer GRN from of single-cell RNA sequencing (scRNA-seq) data. In particular, we developed a modified version of Neural Relational Inference (NRI) that is able to explicitly model the regulatory relationships between genes. Our model takes the form of a variational auto-encoder, in which the latent code represents the underlying GRN and the reconstruction is based on graph neural networks. Results show that GRNRI achieves comparable or better performance on most of benchmark datasets compared with the state-of-the-art methods.
## GRNRI Model
![image](https://user-images.githubusercontent.com/25415940/198849340-172c06be-240f-4547-8ae9-c2434e5ca07a.png)

# GRNRI
The reconstruction of gene regulatory networks from gene expression data is a challenging problem in systems biology. Here we propose GRNRI: an unsupervised model that learns to infer GRN from scRNA-seq data. In particular, we developed a modified version of Neural Relational Inference (NRI) that is able to explicitly model the regulatory relationships between genes. Our model takes the form of a variational auto-encoder, in which the latent code represents the underlying GRN and the reconstruction is based on graph neural networks. Results show that GRNRI achieves comparable or better performance on most of benchmark datasets compared with the state-of-the-art methods.

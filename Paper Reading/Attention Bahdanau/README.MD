# Original Attention Paper

Paper: [NEURAL MACHINE TRANSLATION BY JOINTLY LEARNING TO ALIGN AND TRANSLATE](https://arxiv.org/abs/1409.0473)

The main reason to invent attention was that the last hidden state of the encoder, in the encoder-decoder network, was a bottleneck for the decoder whereby adequate information was not accessible to the decoder.

The attention mechanism allows the network to conveniently *soft-search* the source sentence for parts that are relevant for predicting a word.

$c_i = \sum{\alpha_{ij}j_j}$

Context vector $c_i$ is the input for the decoder hidden state which is recomputed by the alignment model at each timestep where

$a_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}{exp{(e_{ik})}}}$

and

$e_{ij} = v_a^Ttanh{(W_as_{t-1} + U_aj_j)}$. 

$h_j$ is the j-th hidden vector in the encoder hidden state, $v_a^T$ is a linear transformation.
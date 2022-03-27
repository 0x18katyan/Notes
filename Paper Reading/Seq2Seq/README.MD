# Reading Notes

Paper Link: [Sequence to Sequence Learning with Neural Networks
](https://arxiv.org/abs/1409.3215)

## Introduction

Seq2Seq uses multi-layered LSTM to map input sequence to a vector of fixed dimensionality which is then fed into another LSTM for decoding the sequence. The authors reverse the order of the source sentence so that the proximity of the source and target sentences is close which better optimized the model.

## Model

Recurrent Neural Network is a generalization of the Deep Neural Network to sequences. RNNs are useful when there is a one-to-one alignment of inputs to outputs. Before Seq2Seq, there wasn't an agreed way to align two sequences that had variable lengths with complicated and non-monotonic relationships.

A general strategy is to map the input sequence into a fixed-dimension vector using RNN and, then map the vector to another target sequence with another RNN.

An RNN is used to estimate the conditional probability $p(y_1...y_T|x_1...y_{T\prime})$ where $(x_1...x_T)$ is an input sequence and $(y_1...y_{T\prime})$ is the output sequence where the length $T$ may not be equal to ${T\prime}$.

The last state of the encoder RNN is used as the initial state for the decoder RNN.


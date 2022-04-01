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

```Python
for i in (target_len):  ##or target_max_len if it is batched
    decoder_outputs, decoder_hidden = RNN(decoder_input, decoder_hidden)
    
    decoder_input = log_softmax(decoder_hidden, dim = -1) ## when decoder_outputs is shape [batch, 1, vocab_size]

    decoder_input = decoder_input.argmax(dim = -1) ## take the index of the maximum value

    decoder_input = decoder_input.squeeze(1) ## Remove the timestep dimension from the hidden

    ##decoder_input shape is [batch, 1]

    predicted_tensor[i] = decoder_input

```

The decoder input is run through an embedding layer which changes the `[batch, 1]` input to embedded `[batch, embedding_size]`. The decoder_hidden is passed from the previous timestep to the next timestep unchanged for seq2seq (in architectures without attention or any other training shenanigans).

Sometimes teacher forcing is also used for teaching the decoder what to output based on a correct previous input.

```Python
teacher_forcing = 0.5

output = log_softmax(decoder_hidden, dim = -1)
output = output.argmax(dim = -1).squeeze(1)
target[i] = output

## for next iteration
if torch.rand(1) <= teacher_forcing: ## generate a probability from a uniform distribution of [0, 1] and do if it satisfies the teacher_forcing threshold
    ##use the correct prediction at t as the next input
    decoder_input = target[i+1] ## Because we are predicting $y_{t+1}$ at $y_t$.

else:
    decoder_input = output ##If not teacher forcing then just use previous output as input
```
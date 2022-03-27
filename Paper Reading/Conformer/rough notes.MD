# Notes

## Convolutions

Depth-wise convolution is with ($k*k*1$).

Point-wise convolution is with ($1*1*k$).

## Shallow Fusion

### Introduction

Shallow Fusion is a technique where the decoder part of the sequence-to-sequence ASR model is merged with an external Language Model to improve predictions.

### Why Shallow Fusion?

Most of the sequence-to-sequence models are trained in an End-to-End fashion such that they only *see* audio-text-pairs which are already transcribed and present in the training data. This introduces a problem where rare words, like nouns, are not present in the training data and as such can be misclassified during inference. To solve this, an external Language Model, which is usually pre-trained on a larger text corpus, is integrated with the original ASR model to improve accuracy of the predictions.

Shallow Fusion is only used during inference.

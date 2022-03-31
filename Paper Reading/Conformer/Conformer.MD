# [Conformer Paper](https://arxiv.org/pdf/2005.08100.pdf) Notes

CNNs are better are extracting local features while transformers are better for modeling global interactions. Conformer combines CNNs with Transformers. Conformer significantly outperforms other architectures at Automatic Speech Recognition.

CNNs are good for local connectivity but struggle to tie it up with the global context. Thus, a lof of effort has been expended to eradicate this *weakness* of CNNs by techniques such as Residual Layers.

Uses a decoder to decode the hidden representations from the Conformer Encoder, then uses a Transducer style shallow fusion to fuse outputs from the decoder and another language model to produce final outputs at timestep t.

The Language Model is pre-trained on the training data and represents the conditional probability of $y_t|y_{t-1}$ with $y_{t-1}$ being the previous decoded token, as any other AutoRegressive Language Models.


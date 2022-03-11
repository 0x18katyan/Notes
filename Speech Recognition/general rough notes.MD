# Automatic Speech Recognition (ASR)

## Feature Extraction

Most used feature extraction techniques:

1. Mel-frequency cepstral coefficients (MFCCs)
2. Discrete Wavelet Transform (DWT)

Language Model is an important part of the ASR since it is responsible for generating grammatically correct text.

### Mel Frequency

MFCC feature extraction is composed of the following steps:

1. Window the signal.
2. Apply Discrete Fourier Transform.
3. Logarithm of the magnitude.
4. Convert to a Mel scale.
5. Apply inverse discrete cosine transform (DCT).

## Deep Neural Networks for ASR

CNNs and RNNs have previously been used for ASR while currently Transformer based architectures are delivering great results.

### Recurrent Neural Networks

In Speech Recognition, the information of the future context is equally as important as the past context so BiDirectional RNNs have been more commonly used.

**CTC is an objective function that computes the alignment between the input speech signal and the output sequence of the words.**

## Mel Spectrograms


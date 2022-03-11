# Data Preparation and Augmentation

## Spectrogram Optimizations

Mel Spectrograms can be optimized for performance based on the use-case.

### STFT vs DFT

Discrete Fourier Transform (DFT) is a technique used for generating Fourier Transforms for a signal. Fast Fourier Transform is a technique for efficiently computing DFT. However, FFT ,or DFT, computes the Fourier Transform for the entire signal at once which results in frequencies that are present throughout the signal duration. This approach doesn't take into account any change in frequencies at specific instants or segments in the time.

Short-time Fourier Transform enables us to view the variations in frequency over time. It segments the signal using a sliding window technique and then computes FFT for each of those segments and then combines them in the end. This provides us with a single view of variation in frequencies over time.


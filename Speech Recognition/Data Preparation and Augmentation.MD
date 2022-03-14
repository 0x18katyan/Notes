# Data Preparation and Augmentation

## Spectrogram Optimizations

Mel Spectrograms can be optimized for performance based on the use-case.

### DFT

Discrete Fourier Transform (DFT) is a technique used for generating Fourier Transforms for a signal. Fast Fourier Transform is a technique for efficiently computing DFT. However, FFT ,or DFT, computes the Fourier Transform for the entire signal at once which results in frequencies that are present throughout the signal duration. This approach doesn't take into account any change in frequencies at specific instants or segments in the time.

### STFT

Short-time Fourier Transform enables us to view the variations in frequency over time. It segments the signal using a sliding window technique and then computes FFT for each of those segments and then combines them in the end. This provides us with a single view of variation in frequencies over time.

The splits are performed in the time domain and the frequency domain. The whole frequency range is divided into equally spaced bands in the Mel scale. Then, for each of the frequency band, the Amplitude, or energy, is calculated.

#### Example

If we have a 60-seconds clip with frequency range of 0-1000Hz, then the Mel Spectrogram algorithm steps include:

1. Choosing windows that split the signal into 5 sections along the time-axis.

2. Splitting the frequency into 10 bands (0 - 100Hz, 100 - 200Hz, 200 - 300 Hz... 900 - 1000Hz).

3. Apply Fourier Transforms. (*Need Clarity*)

The output from the steps `1` & `2` would be an array of shape `(10, 5)`, (band-sections, time-sections). Each of the 5 column represents the FFT for a single time section while each of the 10 rows represents the Amplitude values for a frequency band.

The first column, that is the Fourier Transform for the first time section, has 10 rows:

- The first row is the Amplitude for the first frequency band 0-100Hz.
- The second row is the Amplitude for the second frequency band 100-200Hz.
- ... and so on.

**Each column in the array is a column in the Mel Spectrogram image.**

#### Hyperparameters for Mel Spectrogram

##### Frequency Bands

- **fmin** - minimum frequency
- **fmax** - maximum frequency
- **n_mels** - number of frequency bands (i.e. Mel bins.). This will be the **height** of the Mel Spectrogram.

##### Time Sections

- **n_fft** - window length for each time section.
- **hop_length** - could be interpreted as *stride* from CNN. Width of Spectrogram = $\frac{total\_number\_of\_samples}{hop\_length}$.

### MFCC - Mel Frequency Cepstral Coefficients (for Human Speech)

MFCC is a compressed representation of the frequency bands from the Mel Spectrogram. It selects the frequency bands that are related the most to Human Speech. This process helps in reducing the dimensions for the feature matrix.

## Data Augmentation

### Spectrogram Augmentation

- Frequency mask - randomly mask out the range of consecutive frequencies by adding horizontal bars on the spectrogram.

- Time mask - same as above except we block out the time ranges by adding vertical bars.

**Frequency is the y-axis and time is the x-axis on the Spectrogram while the color indicates the Amplitude of the frequencies.**

### Raw Audio Augmentation

1. Time Shift - Shift the audio left or right randomly using `SpecAugment`.

    - Good for audio with no temporal dependency, i.e. the order of the audio doesn't matter so that it can wrap around. Example: sound of waves.

    - Human Speech - When order doesn't matter, the gaps are filled with silence or random white noise.

2. Pitch Shift - Randomly modify the frequency of parts of the sound.

3. Time Stretch - Randomly slow down or speed up the sound.

4. Add Noise -Add random values to the sound.

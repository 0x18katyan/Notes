# Notes

Sound Classification is how we classify sounds and predict a category for that sound.

The 'MNIST' of Audio Classification is when sound files are loaded and converted into spectrograms then a CNN + Linear Classifier Model is used to predict which category the Audio belongs too. Example categories: dog barking, train station.

Don't load all the audio data into the memory if the machine doesn't allow it, rather use batch loading and processing as with other algorithms to feed batches while training.

Sampling rate is how many data points are there in 1 second of audio. For example: 1 hz to 1 data point per second, which ,by the way, is not suitable at all for sound recording. The mostly used sampling rates are 44100Hz and 48000Hz which mean 44100 and 48000 data points for 1 second of audio.

Steps:
    1. Prepare Training Data: Load Data, Load Labels for Data.
    2. Preprocess Training Data: Read Audio, Standardize Channels, Standardize Sample Rate, Apply Padding or Truncate, Apply Augmentation,

Read Audio: Read from file and convert to Tensor
    Using torchaudio:
        signal, sampling_rate = torchaudio.load(audio_file)

Convert to two channels: Needed to standardize among different training files. 1 channel is mono and 2 channels is stereo.

```Python
    If num_channels == required_num_channels:
        return
    Else if num_channels == 1:
        return signal[:1, :] ## Assuming that the first dimension is the channel dimension.
    Else:
        return torch.cat([signal, signal]) ##Just copy the signal to the second channel dimension.
```

Resample using torchaudio:

```Python
        resampled_signal = torchaudio.transforms.Resample(sampling_rate, new_sampling_rate)(signal[:1, :])

        if num_channels > 2:
            resampled_signal_2 = torch.audio.transforms.Resample(sampling_rate, new_sampling_rate)(signal[1:, :])

            ##Merge both of them

            resampled_signal = torch.cat([resampled_signal, resampled_signal_2])
            
            return resampled_signal
```

## Data Preprocessing

### Padding or Truncate

Pad or Truncate Signals to make them same length as other signals. Use `0` for padding.

### Time Shift

Time Shift allows us to augment data for generating more training samples for the model.

### Mel Spectogram

[Source: Apple Blog Post](https://developer.apple.com/documentation/accelerate/computing_the_mel_spectrum_using_linear_algebra)

- The mel scale is a scale of pitches that humans perceive to be equidistant from each other.

- As frequency increases, the interval, in hertz, between mel scale values (or simply mels) increases.

- The mel scale is based on the comparison between pitches.

- The mel spectogram converts the values in Hertz to the Mel Scale.

- Mel spectogram is important due to the several nuances of human speaking like accents and pitch change reflecting emotional states.

- In mel spectogram, the distance between (1000Hz, 2000Hz) and (2000Hz, 4000Hz) is the same.

- Mel Spectogram computes the outputs by multiplying the frequency-domain values by a filter bank.

### Fourier Transform

Fourier Transform allows us to decompose a signal into individual frequencies and the frequency's amplitudes. It converts signal from the time domain into the frequency domain. The result is a spectrum.

For non-periodic signals, the signal can be windowed then transformed into the spectrum. This is called short-time Fourier Transform. The FFT is applied to overlapping windows of the signal and the result is called a spectogram.

Multiply the time-domain values by the Hann window to reduce spectral leakage.

### Transform from Multi Channel to Single Channel

$mono = \frac{chan_1 + chan_2...chan_n}n$

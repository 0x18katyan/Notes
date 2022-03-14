# Audio Deep Learning

## References

Articles :

1. [Audio Deep Learning Made Simple Part 1](https://towardsdatascience.com/audio-deep-learning-made-simple-part-1-state-of-the-art-techniques-da1d3dff2504)

2. [Audio Deep Learning Made Simple Part 2](https://towardsdatascience.com/audio-deep-learning-made-simple-part-2-why-mel-spectrograms-perform-better-aad889a93505)

3. [Audio Deep Learning Made Simple Part 3](https://towardsdatascience.com/audio-deep-learning-made-simple-part-3-data-preparation-and-augmentation-24c6e1f6b52)

## What is sound?

Sound signal is produced by variations in air pressure. Sound signals are often periodic and each wave has the same shape. The height is the intensity of the sound and is called the amplitude.

The time taken for a signal to complete one full wave is the period. The number of waves in a signal within a second is called the frequency. The frequency is also the reciprocal of the amplitude and it's unit is Hertz.

Sounds in our daily life are not as simple as one recurring wave but consists of different frequencies that forms composite signals with complex repeating patterns.

All sounds that humans hear consist of waveforms. The human ear is able to differentiate between sounds using the *quality* of the sound which is known as **timbre**.

### How do humans hear frequencies?

Pitch is a subjective impression of the frequency. A high-pitched sound has a higher frequency than a low-pitched sound. Research indicates that humans are more sensitive to differences in lower frequencies than higher frequencies.

For example, we are biologically attuned to perceive that the distance between pairs 100Hz and 300Hz is greater than the distance between 1500Hz and 3000Hz. And, it is harder to distinguish the difference between sounds as the frequency increases.

#### Mel Scale

The Mel Scale was invented to address this non-linearity in pitch perception by our brain. It is a scale of pitches such that frequencies represented by the same unit in the mel scale are equal in pitch distance according to human perception.

**Only frequencies in the range of [0, 1] kHz can be transformed to the Mel-scale**; the remaining frequencies are considered to be logarithmic.

### How do humans hear amplitude?

Humans process amplitudes same as frequencies. The ability to discern changes in amplitude decrease with the increase in amplitude; we lose the ability to distinguish between loudness as the loudness of a sound increases. The Decibel Scale accounts for this phenomenon in amplitude as Mel Scale accounts for the same in frequencies.

#### Decibel Scale

On the Decibel Scale, 0 dB is total silence. From 0 the measurement units increase exponentially. An increase of 10dB in the decibel scale is equal to a 10-fold increase in the "volume" of the sound. For example, 10 dB is 10 times louder than 0 dB and 30 dB is 1000 times louder than 10dB. Any sound more than 100 dB is not bearable to human ears.

## Digitally Representing Sounds

A digital sound wave consists of amplitudes measured at a constant time interval. Each such measurement is called a sample, and the number of times such a sample is measured within a second is called a sampling rate.

### Spectrum

A signal consisting of distinct frequencies can be expressed as the sum of the those frequencies. A spectrum is used to denote the source frequencies that are present in a signal. The lowest frequency in a signal is called the fundamental frequency and the frequencies that are integer multiples of the fundamental frequency are called harmonics.

### Time Domain vs Frequency Domain

Amplitude against Time is the way of representing a waveform in the time domain while Amplitude against Frequency is the way of representing a waveform in the Frequency domain.

The Spectrum represents the plot of Amplitude vs Frequency at a given time.

### Spectrograms

Spectrogram of a signal is the collective spectrum of the signal as it progresses through time. While plotting a spectrogram, brighter color indicates more amplitude or strength of the frequency at that instant in time.

Spectrograms are generated using Fourier Transforms that decomposes any signal into its constituent frequencies.

Summary: Spectrogram is a compact representation of an audio signal that helps in capturing features of the signal.

### Generating a Spectrogram

Spectrograms of a signal are generated using Fourier Transforms that decomposes a signal into its constituent frequencies and displays the amplitude for each of them.

Spectrogram segments the duration of the sound signal into smaller time frames and then applies Fourier Transform to each segment individually for obtaining the frequencies in that particular segment. It them combines the results of the Fourier Transform of all the segments into one single plot.

The spectrogram plots Frequency in the y-axis vs Time in the x-axis. Different colors are used to denote the amplitude of each frequency with brighter colors indicating more strength.

#### Mel Spectrograms: Mel Scale fused with Decibel Scale

Mel Spectrograms are the results of using Mel Scale for representing frequencies and Decibel Scale for representing amplitude of a signal.

#### Mel Frequency

MFCC feature extraction is composed of the following steps:

1. Window the signal.
2. Apply Discrete Fourier Transform.
3. Logarithm of the magnitude.
4. Convert to a Mel scale.
5. Apply inverse discrete cosine transform (DCT).


## Uses of Deep Learning for Audio

- Audio Classification - Detect specific sounds like a fire alarm.

- Audio Separation - Separate desired sound from a mixed audio clip like speech volume boosting during phone calls.

- Audio Segmentation - Used to highlight specific parts of audio such as a chorus part in a song.

- Music Genre Classification - Classify music into different genres based on the sounds of the music.

- Music Generation and Music Transcription - Generate music based on transcriptions or annotate a song.

- Voice Recognition - Identify gender or age of the speaker using their voice. Or identify their mood using their voice.

- Speech to Text and vice versa - Convert audio to text and convert text to audio.

### Processing Audio for Deep Learning

Before machine learning, digital signal processing was used to hand craft features for any audio related tasks. For example, phonetics concepts were used to extract phonemes which would be used to analyze audio signals. These types of traditional methods required tons of domain expertise and were hard to tune for performance.

Modern deep learning methods do not rely on hard crafted features, and standard data preparation techniques can be used for processing audio. A common approach is to first convert the audio signals into spectrograms and then use CNNs to perform any sort of classification task. Obviously this is a general example and state-of-the-art Automatic Speech Recognition models are a wee bit different but the point is audio signal processing is a lot more accessible due to the rise of machine learning.

#### Feature Extraction

Most used feature extraction techniques:

1. Mel-frequency cepstral coefficients (MFCCs)
2. Discrete Wavelet Transform (DWT)

Language Model is an important part of the ASR since it is responsible for generating grammatically correct text.

#### Loading an audio file

A Sound Wave can be visualized in Python using librosa by:

```Python
import librosa
import librosa.display
import matplotlib.pyplot as plt

AUDIO_FILE = './someaudio.wav'
samples, sampling_rate = librosa.load(AUDIO_FILE, sr = None)


plt.figure(figsize=(20,8))
librosa.display.waveplot(samples, sr = sampling_rate)
```

and the audio can be played in a Jupyter Notebook using:

```Python
from IPython.display import Audio
Audio(AUDIO_FILE)
```

When an audio signal data is loaded, the data is decompressed from the file into an array. An audio clip with sampling rate of 16000 and length of 1 second will produce an array of length 16000. The array will only contain amplitude values.

Bit-depth is used to gauge the range of amplitude measurements for each sample. A bit-depth of 16 means that amplitude can vary from 0 to 65535 ( $2^{16}$  - 1). Higher bit-depth means the audio will contain more information that closely resembles the original sound of the recording.



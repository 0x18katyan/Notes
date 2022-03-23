# [Sequence Modeling with CTC](https://distill.pub/2017/ctc/)

Audio clips and transcriptions are not aligned in datasets and it is prohibitively expensive to do so. This problem is persistent in many other domains such as handwriting recognition and extracting information from video frames.

For any sequential input $X = [x_1, x_2, x_3...x_n]$, such a audio waveforms or feature vectors of a video frame, and an output $Y = [y_1, y_2, y_3...y_n]$, there exists 2 main problems for directly mapping each input to output:

1. Variable Lengths of $X$ and $Y$. Most of the cost functions are designed for 1-to-1 mapping of *predicted* and *truth* values which is not possible for all sequential data like audio; whereby, multiple input values in $X$ may correspond to a single output value in $Y$.

2. Since it is costly to manually annotate data such as audio or video, exact mappings of $X$ to $Y$ **are not** available.

CTC outputs an output distribution over all possible $Y$ for a given $X$ at timestep $t$. The output distribution can be used to infer the most probable output in $Y$ for the input $X$.

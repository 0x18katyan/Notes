# Squeeze and Excitation Networks

Reference: [Squeeze and Excitation Networks](https://blog.paperspace.com/channel-attention-squeeze-and-excitation-networks/)

Original Paper: [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)

Squeeze and Excitation Networks' main contribution is a channel attention mechanism. It is a simple and efficient module that can be added to any existing CNN architecture to improve the accuracy of the model with minimal computational overhead.

Some channels might not have the same importance as other channels for an image as they are derived by applying convolutional kernels to a patch and extract different features and, during training, a certain feature might be more important than another feature for that task or image.

Thus, it is intuitive to apply an attention weight to different channels based on their importance before propagating them to the subsequent layer.

## Architecture

The Squeeze-and-Excite block (SE-block) consists of 3 components:

1. Squeeze Module
2. Excitation Module
3. Scale Module

### Squeeze Module

Reduce the dimension of the feature maps (Channels x Height x Width). This is obtained by applying Global Average Pooling to the incoming data from the previous layer.

This is essential because the number of channels increase as the data propagates deeper into the network, and the number of channels is directly proportional to the number of parameters. Without reducing the `size` of the feature maps, the parameter size of CNN will significantly increase with the addition of the excitation module.

Thus, a feature decomposer is needed to reduce the size of the channels. The authors experimented with Global Average Pooling and Global Max Pooling and found that Global Average Pooling performed better in their experiments and thus opted for Global Average Pooling as the decomposer.

- Max pooling preserves the most activated pixels but it can be noisy and doesn't contain information from its neighbors.

- Average pooling doesn't preserve the information about the most activated pixels but it constructs the average of the area.

Global Average Pooling is applied to obtain a output tensor of shape ($C *1* 1$) from an input of ($C *H* W$). This results in each of the feature maps to be reduced to a single value.

### Excitation Module

During the excitation step, the output from the squeeze module is propagated through a Multi Layer Perceptron (MLP) Network with a bottleneck layer. The idea is similar to autoencoders and lets us extract important information contained in the input data.

The bottleneck layer is determined by  a parameter `r` which is the reduction ratio for the layer. Essentially, the number of neurons in the bottleneck layer $= \frac {C}{r}$, where *C* is the number of channels and *r* is the reduction ratio.

### Scale Module

During the scale module section, the *excited* output from the excitation module, is scaled using an activation function to obtain the weights for each of the original feature maps. The authors experimented with different activation functions like Sigmoid, ReLu, and Tanh and found Sigmoid to be the best performing of the lot.

Then *activated* output is applied directly to the feature maps by performing a broad-casted element-wise multiplication.

### Summary

To summarize, the SE block computes the importance vector for the feature maps by:

1. *Squeezing* feature maps into a singular value.
2. Propagating the reduced feature maps through a MLP to extract excited tensor values of the feature maps.
3. Apply non-linearity to the *excited* values from the excitation module.

## Application to Architecture

The SE block can be placed in different positions and settings for an architecture. The authors found best results, in a residual layers based architecture, when they applied SE block to an incoming feature maps input independently and then apply it to the output from an adjacent Residual layer.

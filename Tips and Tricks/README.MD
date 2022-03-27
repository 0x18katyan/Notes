# Tips and Tricks

## Machine Learning Engineering

- Be sure to check the maximum length of the sequence inputs to ensure that there aren't any anomalies that might cause the data pipeline to pad unnecessarily to accommodate the anomaly. Instead, try to create a separate dataset from the larger length sequences that might be better as development validation set.

- Oscillating losses can have several reasons, some are:

    1. High learning rate - A higher learning rate means the model is being unable to optimize properly to the local minima. Try *lowering* the learning_rate either by using a **scheduler** or starting with a low learning rate.

    2. Exploding gradients - Exploding gradients can be another reason for the fluctuating loss. This can be neutralized by **clipping gradients** either by value or by norm.

    3. Small batch size - Small batch size means that the gradients are not representative of the general distribution of the outputs. If increasing batch size is not possible, try **gradient averaging**.
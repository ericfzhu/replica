## Introduction
The original breakout convolutional neural network (CNN) introduced by Alex Krizhevsky et al. in 2012, which achieved SOTA top-1 and top-5 error rates of 37.5% and 17.0% on the 2010 ImageNet dataset respectively. This model featured a much deeper neural network architecture, the use of ReLU (Rectified Linear Unit) activation functions, and a dropout regularization technique to prevent overfitting.

## Architecture
ReLU activations ($f(x) = \max(0, x)$) are used in favor of the standard tanh activation function ($f(x) = \tanh(x)$), as the ReLU-based models are able to be trained several times faster than the tanh-based models.

Local response normalization is given by the expression 

$$b_c = a_c \left(k + \frac{\alpha}{n} \sum_{c'=\max(0,c-n/2)}^{\min(N-1,c+n/2)} (a_{c'})^2\right)^{-\beta}$$ 

where the hyper-parameters are set as $k = 2, n = 5, \alpha = 10^{-4}$ and $\beta = 1.5$. Using response normalization reduces the top-1 and top-5 error rates by 1.4% and 1.2% respectively.

Pooling layers are used to summarize the feature maps of the convolutional layers. A pooling layer can be thought of as consisting of a grid of pooling units spaced $s$ pixels apart, each summarizing a neighborhood of size $z \times z$ centered at the grid unit's location. Overlapping pooling is used with $s = 2$ and $z = 3$, which reduces the top-1 and top-5 error rates by 0.4% and 0.3% respectively.

Reponse normalization layers follow the first and second convolutional layers, and are followed by a pooling layer. The fifth convolutional layer is also followed by a max-pooling layer.

![Model diagram](models/alexnet/image1.jpg)

## Training


## Results

WIP
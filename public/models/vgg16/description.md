## Introduction
Simonyan and Zisserman, 2014 introduced the importance of depth in the architectural design of the convolutional neural network (CNN). This is achieved by fixing the other parameters of the architecture, and steadily increasing the depth of the network by adding more convolutional layers by using small ($3 \times 3$) convolutional filters (kernel sizes) in all layers.

## Architecture
Using smaller kernel sizes allows the use of more ReLU layers, which makes the decision function more discriminative. This also has the benefit of reducing the number of parameters in the network, as a 3-layer $3 \times 3$ convolutional layer ($3(3^2C^2) = 27C^2$) stack has fewer parameters than a single $7 \times 7$ convolutional layer ($7^2C^2 = 49C^2$) where $C$ is the number of channels.

Max-pooling is performed with a kernel size of 2 and stride of 2.

All hidden layers use ReLU activation functions.

None of the networks except one of the proposed configurations contain Local Response Normalization (LRN) layers, which the authors claim does not improve the performance on the ILSVRC dataset, but leads to increased memory and computational requirements.

The implemented model is configuration D of the proposed architecture.
![Model diagram](models/vgg16/image1.jpg)

## Training
Waiting for datasets to be downloaded. Will be trained on the [ILSVRC 2010](https://www.image-net.org/challenges/LSVRC/2010/index.php) dataset for object classification and  [ILSVRC 2012](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset for object localization and object detection.


## Results

WIP
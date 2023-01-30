# Standalone model
<!--
ResNet architecture: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8
ResNet expects input images of size: 224x224
-->
In order to demonstrate a caption-based explainable AI method, a model to be explained is needed. This chapter introduces the trade-off between the interpretability and the accuracy of different machine-learning models, the final choice for the architecture of the biased standalone model and assesses its performance on the custom dataset.

## Interpretability vs. accuracy
<!-- Which models are available to choose from? -->
\*@fig:interpretability_vs_accuracy illustrates the tradeoff between the interpretability and accuracy of different machine learning models. The weights of simple linear regression models are directly interpretable, but the accuracy could be better for complex problems. Decision trees offer an excellent intrinsic explanation of their prediction by design. Support vector machine classifiers use the kernel trick to find a separating hyperplane in a higher dimensional space. Transforming this hyperplane back to a lower dimensional space results in a hard-to-interpret non-linear decision boundary. Random forests consist of many interpretable decision trees, but interpreting the result is difficult due to the randomness involved and their voting process. There is a trend to use deeper and deeper neural networks with millions and billions of tuneable parameters, which make them very successful function approximators in terms of accuracy but challenging to interpret. This work focuses on developing a caption-based explainable AI method for a neural network in a machine-vision problem setting.

![Increasing accuracy comes at the cost of decreasing interpretability for linear regression, decision trees, support vector machines (SVM), random forests and neural networks. [[@interpretability_vs_accuracy]](#references)](source/figures/Model-interpretability-vs-accuracy.png "Model interpretability vs. accuracy."){#fig:interpretability_vs_accuracy width=60%}
 
## Model selection
<!-- Why resnet? How does it work/look like? -->
<!-- ResNet identity mapping: https://medium.com/deepreview/review-of-identity-mappings-in-deep-residual-networks-ad6533452f33
Batch norm: https://towardsdatascience.com/batch-norm-explained-visually-how-it-works-and-why-neural-networks-need-it-b18919692739
-->
As depicted in \*@fig:resnet_imagenet, AlexNet achieved the first outstanding top 5 classification error of 16.4% on the ImageNet challenge in 2012. Using the rectified linear unit (ReLU) activation function and using different convolution kernel sizes bypasses the vanishing gradient problem. A considerable improvement brought the VGGNet in 2014 with a top 5 error rate of 7.3%. Using smaller convolution kernels of the same size is the most impactful change. This improvement leads to less trainable parameters, enables faster learning and tends to be more robust to overfitting.

![Relative image classification error (Top 5) in percent of different machine learning models on the ImageNet challenge. A human's relative image classification error (Top 5) is about 5%. [[@resnet_imagenet]](#references)](source/figures/resnet_imagenet.png "Imagenet classification error top 5."){#fig:resnet_imagenet width=90%}

The residual neural network (ResNet) architecture was a real breakthrough in 2015 because it was the first model which achieved "super-human" performance in the ImageNet challenge with a top 5 error rate of 3.6%. Therefore, a ResNet-50 model is the model of choice to use as the standalone model for this work. The ResNet architecture does not bypass the vanishing gradient problem like AlexNet but solves the problem by introducing shortcut connections that perform identity mappings. Convolutional layers enclosed by a shortcut connection are called residual blocks. As shown in \*@fig:residual_blocks, different types of residual blocks are available.

![There are different types of residual blocks available. The ResNet-34 uses the "basic" residual block on the left. The ResNet-50/101/152 use the "bottleneck" residual block on the right. [[@he_deep_residual_learning]](#references)](source/figures/residual_blocks.png "."){#fig:residual_blocks width=100%}

<!-- https://iq.opengenus.org/resnet50-architecture/ -->
These innovative residual blocks solved the vanishing gradient problem and enabled deeper and more powerful neural networks. There are various well-known residual neural networks, like ResNet-18, ResNet-34, ResNet-50, ResNet-101 and ResNet-152. The number after ResNet in the type denotes the number of neural network layers in the architecture, as shown in \*@fig:resnet_architecture. This work uses the ResNet-50 with 50 neural network layers consisting of 49 convolutional layers and one fully connected layer. There is one additional max-pooling and one average-pooling layer, but these are not counted towards the number of neural layers since they are pooling layers.

![Basic structure of a residual neural network (ResNet). The ResNet-50 model is divided into five stages and consists of 50 neural network layers. [[@resnet_architecture]](#references)](source/figures/resnet50_architecture.png "Architecture of a residual neural network (ResNet)."){#fig:resnet_architecture width=100%}

\*@fig:resnet50_configuration denotes the  exact configuration of the ResNet-50 with ~23 million trainable parameters. The number next to the brackets denotes the number of stacked building blocks. Downsampling is not just done by pooling layers but mainly by conv3_1, conv4_1 and conv5_1 with a stride of 2.

<!-- 
Best resnet explanation:
https://cv-tricks.com/keras/understand-implement-resnets/#:~:text=Architecture%20of%20ResNet%2D50&text=For%20the%20sake%20of%20explanation,%C3%973%20kernel%20sizes%20respectively. -->
![Model architectures of ResNet-18, ResNet-34, ResNet-50, ResNet-101 and ResNet-152. [[@he_deep_residual_learning]](#references)](source/figures/resnet50_configuration.png "."){#fig:resnet50_configuration width=100%}

## Performance
<!-- accuracy on train/validation (good) and test (biased) -->
A working explainable artificial intelligence (XAI) method should be able to reveal problematic models. A ResNet-50 model is used to create such a biased standalone model. A ResNet-50 model offers a good tradeoff between its performance on a large number of tasks and its complexity. Additionally, a ResNet-50 model is more interpretable than its predecessors, as described in \*@sec:network-dissection.

Using a pre-trained ResNet-50 model accelerates the training progress. The model in use is pre-trained on the ImageNet dataset (ILSVRC 2012) with 1000 classes, ~1.2 Mio training images and 50 thousand validation images. [@imagenet] The process of replacing and training the final classification layers is called transfer learning. The hyperparameters used are shown in \*@tbl:biased_standalone_hyperparam_table.

|Hyperparameter     | Value
|-                  | -           
|Batch size         | 128
|Learning rate      | 0.0000001
|Number of epochs   | 30
|K-folds            | 5
Table: Hyperparameters used to train the standalone model. {#tbl:biased_standalone_hyperparam_table}

Using the dataset introduced in \*@sec:dataset for the binary classification task to distinguish between two digits results in a low bias, low variance model. The learning curves for the training, validation and test datasets are documented in the \*@fig:performance_biased_without_test_fool. Everything looks satisfactory from a developer's point of view. The usual reaction to this kind of figure is: The model is ready to be deployed!

![This figure illustrates the low bias, low variance learning progress of the transfer learned ResNet-50 model on the biased color MNIST training, validation and test datasets during the model development process.](source/figures/performance_biased_without_test_fool.png "Training, validation and test learning curves from standalone ResNet-50 on custom MNIST dataset for binary classification."){#fig:performance_biased_without_test_fool width=50%}

Simulating the deployment of this standalone model by exposing it to the real-world environment, the accuracy decreases dramatically! Evaluating the performance on the real-world dataset converges towards an accuracy of ~50%, as shown in \*@fig:performance_biased_with_test_fool. This situation demonstrates that assessing the training, validation and test learning curves is the way to find a high bias or high variance problem during the development process with respect to the dataset at hand. This property could tempt a developer to release the standalone ResNet model too early. However, there might be a covariate shift between the datasets at hand during the development and the real-world data, which prevents the model from learning and executing its designated task. In this situation, the standalone ResNet model performance suffers from a covariate shift between the data available during development (Perfect correlation between colors and value of the digits in the training, validation and test datasets) and the data in the real-world (Digits have random colors in the real-world dataset). This biased standalone model (Focusing on colors to classify the digits) is used throughout this work. This standalone model is ideal for demonstrating the caption-based explainable AI method.

![This figure illustrates the low bias, low variance learning progress of the transfer learned ResNet-50 model on the biased color MNIST training, validation and test datasets during the model development process. Additionally, the real-world curve demonstrates the poor performance simulated in the real-world environment.](source/figures/performance_biased_with_test_fool.png "Training, validation, test and real-world learning curves from standalone ResNet-50 on custom MNIST dataset for binary classification."){#fig:performance_biased_with_test_fool width=50%}

## Theory summary
The core of the idea is that the standalone ResNet model trained on this custom dataset will focus on the undesired correlating feature (Color of the digits) instead of the desired causating feature (Shape of the feature) to classify the digits. This bias leads to high accuracy on the training, validation and test datasets while developing a new model and terrible accuracy on the real-world dataset. The presented XAI method should then be able to reveal the problem using the training, validation and test datasets only!

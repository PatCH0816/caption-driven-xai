# Novel XAI method
Machine learning models learn information from data without relying on pre-determined equations as a model. This property makes it challenging to explain why these models work the way they do. This work presents a new XAI method to support better explanations. The explanation's findings help to improve the machine learning model's robustness. This chapter introduces the initial situation, the network surgery process and the overall objective of this new XAI method.

## Initial situation
Starting with a ResNet-50 model, pre-trained on ImageNet as described in \*@sec:standalone-resnet-model and adapted to the binary classification problem to distinguish between photos of the digit 5 and 8 using the modified dataset as described in \*@sec:dataset. The learning curves on the training, validation and test datasets during the development process indicate a low bias and low variance model, as shown in \*@fig:abstract_1_situation. Without using explainable artificial intelligence (XAI) methods, a developer cannot detect any problem at this point. Peeking at the \*@sec:dataset reveals a color bias in the training, validation and test datasets, leading to a biased standalone model.

![The biased standalone ResNet-50 model consists of an actual image encoder and the fully-connected linear classifier. The training, validation and test curves indicate a low bias, low variance model regarding the dataset at hand during development. It cannot be said with certainty whether the model is biased or not without using XAI.](source/figures/abstract/abstract_1_situation.png "Architecture and training, validation and test curves of the biased standalone model."){#fig:abstract_1_situation width=100%}

## Network surgery
text text text.

![The architecture of the novel XAI method uses the core architecture of CLIP. Using CLIP's text encoder (Purple) and image encoder (Green), the resulting embedding similarities reveal what the CLIP image encoder (Green) is focusing on using captions. The network surgery process allows integration of any standalone model into CLIP, so CLIP can explain what the image encoder (Red) from a standalone model focuses on.](source/figures/abstract/abstract_2_clip.png "Overview of the novel XAI method."){#fig:abstract_2_clip width=100%}

## Objective
text text text.

![The novel XAI method reveals the color feature as a highly correlating bias in the dataset available during development. Removing the color feature using a pre-processor and retraining the model makes the standalone model more robust. The captions from CLIP's section of the new XAI method reveal that the feature relevant to the decision-making process shifts from the color to the shape feature.](source/figures/abstract/abstract_3_xai.png "Comparison between the standalone model with and without the use of XAI."){#fig:abstract_3_xai width=100%}

<!-- 
How transferable are features in deep neural networks?
https://papers.nips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html
-->

<!--
layer switching:
- preserve CLIP embedding space
- transfer decision critical layers from given model to CLIP
-->

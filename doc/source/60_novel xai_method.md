# Caption based explainable AI
Machine learning models learn information from data without relying on pre-determined equations as a model. This property makes it challenging to explain why these models work the way they do. This work presents a new explainable artificial intelligence (XAI) method to support better explanations. The explanation's findings help to improve the machine learning model's robustness. This chapter introduces the initial situation, the network surgery process and the overall objective of this new XAI method.

## Initial situation
Starting with a ResNet-50 model, pre-trained on ImageNet as described in \*@sec:standalone-model and adapted to the binary classification problem to distinguish between photos of the digit 5 and 8 using the custom dataset as described in \*@sec:dataset. The learning curves on the training, validation and test datasets during the development process indicate a low bias and low variance model, as shown in \*@fig:abstract_1_situation. Without using the power of XAI methods, a developer cannot detect any problem at this point. Peeking at the \*@sec:dataset reveals a color bias in the training, validation and test datasets, leading to a biased standalone model. The red-colored components in the figures indicate their relationship to the standalone model.

![The standalone ResNet-50 model (Red) consists of an actual image encoder and the fully-connected linear classifier. The training, validation and test curves indicate a low bias, low variance model regarding the dataset at hand during development. It cannot be said with certainty whether the model is biased without using XAI.](source/figures/abstract/abstract_1_situation.png "Architecture and training, validation and test curves of the biased standalone model."){#fig:abstract_1_situation width=100%}

## Network surgery
<!--
How transferable are features in deep neural networks?
https://papers.nips.cc/paper/2014/hash/375c71349b295fbe2dcdca9206f20a06-Abstract.html
-->
The architecture of the caption-based explainable AI consists of the standalone model as described in \*@sec:standalone-model and the contrastive language-image pre-training (CLIP) as described in \*@sec:contrastive-language-image-pre-training. All active components from CLIP, the standalone model and the network surgery are shown in \*@fig:abstract_2_clip. CLIP consists of two encoders: a text encoder (Purple) and an image encoder (Green). The standalone model consists of an image encoder (Red). The network surgery process merges the properties from the standalone model to be explained into CLIP by swapping similar layers from the standalone image encoder (Red) to the CLIP image encoder (Green). In order to reveal if the standalone model (Red) is either focusing on the colors or shapes of the digits, the following four suitable captions are used:

- a photo of a digit with the value 5
- a photo of a digit with the value 8
- a photo of a red digit
- a photo of a green digit

![The architecture of the caption based explainable AI method uses the core architecture of CLIP. Using CLIP's text encoder (Purple) and image encoder (Green), the resulting embedding similarities reveal what the CLIP image encoder (Green) is focusing on using captions. The highlighted similarity scores (Blue) are the largest ones. The network surgery process allows integration of any standalone model into CLIP, so CLIP can explain what the image encoder (Red) from a standalone model focuses on.](source/figures/abstract/abstract_2_clip.png "Overview of the caption based explainable AI method."){#fig:abstract_2_clip width=100%}

The network surgery process consists of the following three main steps:

- Compute statistics
- Activation matching
- Swapping layers

Each of these three main steps is explained in detail in the next sections.

\noindent
**Compute statistics**  
The statistics of interest are the mean and the standard deviation of all activations.

\noindent\fbox{
    \begin{minipage}{\linewidth}
        \begin{equation}
            \mathbf{A} \cdot \mathbf{B} = \| \mathbf{A} \| \cdot \| \mathbf{B} \| \cdot cos(\theta)
        \end{equation}
        \begin{equation}
            \sum_{n=1}^{N} \mathbf{A}_n \mathbf{B}_n = \sqrt{ \sum_{n=1}^{N} \mathbf{A}_n^2 } \sqrt{\sum_{n=1}^{N} \mathbf{B}_n^2 } \cdot cos(\theta)
        \end{equation}
        \begin{tabular}{l @{ $=$ } l}
            $\mathbf{A}$ & Image embedding vector\\
            $\mathbf{B}$ & Text embedding vector\\
            $ \theta $ & Angle between vectors
        \end{tabular}
    \end{minipage}
}
















\noindent
**Activation matching**  
asdf asdf

\noindent
**Swapping layers**  
asdf asdf



## Inference
After the network surgery, the caption-based explainable AI model consists of CLIP's original text encoder (Purple) and the modified image encoder (Red/Green striped) as shown in \*@fig:network_surgery_result_unbiased. The hypothesis is that the network surgery process can merge decision-critical properties and preserve the CLIP embedding space at the same time. 

![The caption-based explainable AI consists of the original CLIP text encoder (Purple) and the post-network surgery image encoder (Red/Green striped). The highest score of the embedding similarities indicates which caption describes the image the best from the original standalone model (Red) point of view.](source/figures/network_surgery_result_unbiased.png "Post-network surgery caption-based explainable AI model."){#fig:network_surgery_result_unbiased width=100%}

Feeding all images through the caption-based explainable AI model shown in \*@fig:network_surgery_result_unbiased and keeping track of the most significant similarity scores demonstrates if the model focuses either on the colors or on the shape of the digits to classify the images. Suppose there is a high number of most significant similarity scores for captions describing the colors, like "a photo of a red digit." or "a photo of a green digit.". In that case, the color feature is revealed as the most dominant feature to classify the images. The standalone model should not use this undesired color feature but focus on the shape of the digits instead. Therefore, one approach is to implement a pre-processor, which converts color images to grayscale images and retrain the standalone model with the hyperparameters shown in \*@tbl:unbiased_standalone_hyperparam_table to remove its color bias.

|Hyperparameter     | Value
|-                  | -           
|Batch size         | 128
|Learning rate      | 0.00000015
|Number of epochs   | 100
|K-folds            | 5
Table: Hyperparameters used to train the unbiased standalone model. {#tbl:unbiased_standalone_hyperparam_table}

The unbiased standalone model trained on the pre-processed grayscale images and its low bias, low variance learning curves are shown in \*@fig:standalone_grayscale_with_performance.

![The caption-based explainable AI model detects color bias in this case. Using a color-to-grayscale pre-processor removes the color bias. The unbiased standalone ResNet-50 model (Yellow) consists of an actual image encoder and the fully-connected linear classifier. This model is retrained on the grayscale images. The training, validation and test curves indicate a low bias, low variance model regarding the dataset at hand during development.](source/figures/standalone_grayscale_with_performance.png "Architecture and training, validation and test curves of the unbiased standalone model."){#fig:standalone_grayscale_with_performance width=100%}

Reviewing the biased standalone model may have predicted the correct class for the red five as shown in \*@fig:abstract_3_xai, but the caption-based explainable AI model reveals the color bias contained in the custom dataset, which leads to a biased standalone model. This knowledge can be used to implement a pre-processor to remove the informationless color bias from the dataset. Retraining the standalone model on the unbiased grayscale dataset leads to an unbiased standalone model, which can predict an image with a digit with the numeric value 5. Due to the detected and removed bias, the prediction relies on the digit's shape and not on its color.

![The caption based explainable AI method reveals the color feature as a highly correlating bias in the dataset available during development. Removing the color feature using a pre-processor and retraining the model makes the standalone model more robust. The captions from CLIP's section of the new XAI method reveal that the feature relevant to the decision-making process shifts from the color to the shape feature.](source/figures/abstract/abstract_3_xai.png "Comparison between the standalone model with and without the use of XAI."){#fig:abstract_3_xai width=100%}

## Theory summary
The core of the idea is that the caption-based explainable AI should be able to use its network surgery process to merge a standalone model to be explained into CLIP. Using suitable captions, the similarity between the text concepts and image concepts will result in high similarity scores. If these high scores primarily arise for the color descriptions, then the standalone model is color biased. If these high scores primarily arise for the shape descriptions, then the standalone model is focused on the shapes to determine the class of an image.

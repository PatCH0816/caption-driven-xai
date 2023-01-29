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

<!-- https://github.com/CSAILVision/gandissect/blob/master/netdissect/nethook.py -->
The details about these three main steps follow in the next sections. To keep track of all activations during inference of the model, a model wrapped called "InstrumentedModel" is used. This wrapper allows hooking arbitrary layers to monitor or modify their output directly. [@instrumented_model_wrapper]

### Compute statistics 
Feeding the training dataset with the images $\boldsymbol{x}$, as introduced in \*@sec:dataset, into the model $h$ and retaining the activations $\boldsymbol{A}$ of all kernels/units $k$ allows us to compute the statistics of all activations. Assuming the activations of all units $k$ are gaussians, then the mean and standard deviation are suitable measures to describe these distributions.

As explained in \*@sec:standalone-model, there are 49 convolutional layers, one fully connected layer and two pooling layers in the ResNet-50 standalone model. The file "./3_miscellaneous/model_architectures/standalone_resnet50.txt" describes the exact architecture of the standalone model. Each convolutional layer has a specific number of convolutional kernels/units. The number of kernels/units $k$ available for swapping in the standalone model is 22'720.

As explained in \*@sec:contrastive-language-image-pre-training, CLIP's image encoder is a modified ResNet model. There are two additional convolutional layers in the first stage of the model. Therefore, there are 51 convolutional layers, one fully connected layer and two pooling layers in CLIP's image encoder model. The file "./3_miscellaneous/model_architectures/clip_resnet.txt" describes the exact architecture of the CLIP model. Retaining the activations from the last layer of each of the five stages of the CLIP image encoder only allows to limit of the computational power needed to an absolute minimum. Each layer has a specific number of kernels/units. The number of all kernels $k$ available for swapping in the last layer of the four last out of five stages in the CLIP image encoder is 3'840.

The mean $\boldsymbol{\mu}_{kl}$ and the standard deviation $\boldsymbol{\sigma}_{kl}$ for each kernel/unit $k$ in the mentioned layers are computed as follows:

\noindent\fbox{
    \begin{minipage}{\linewidth}
    
        \begin{equation}
            C(\boldsymbol{x}) \Rightarrow \boldsymbol{A}^C_{k}
        \end{equation}

        \begin{equation}
            S(\boldsymbol{x}) \Rightarrow \boldsymbol{A}^S_{k}
        \end{equation}

        \begin{equation}
            \boldsymbol{\mu}^C_{k} = \frac{1}{N^C_{k} \cdot M^C_{k}} \sum_{i=1}^{N^C_{k}} \sum_{j=1}^{M^C_{k}} \boldsymbol{A}^C_{kij}
        \end{equation}

        \begin{equation}
            \boldsymbol{\mu}^S_{k} = \frac{1}{N^S_{k} \cdot M^S_{k}} \sum_{i=1}^{N^S_{k}} \sum_{j=1}^{M^S_{k}} \boldsymbol{A}^S_{kij}
        \end{equation}

        \begin{equation}
            \boldsymbol{\sigma}^C_{k} = \sqrt{\frac{1}{N^C_{k} \cdot M^C_{k}} \sum_{i=1}^{N^C_{k}} \sum_{j=1}^{M^C_{k}} (\boldsymbol{A}^C_{kij} - \boldsymbol{\mu}^C_{k})^2}
        \end{equation}

        \begin{equation}
            \boldsymbol{\sigma}^S_{k} = \sqrt{\frac{1}{N^S_{k} \cdot M^S_{k}} \sum_{i=1}^{N^S_{k}} \sum_{j=1}^{M^S_{k}} (\boldsymbol{A}^S_{kij} - \boldsymbol{\mu}^S_{k})^2}
        \end{equation}

        \begin{tabular}{l @{ $=$ } l}
            $\boldsymbol{A}$ & Activation maps\\
            $C$ & CLIP image encoder\\
            $k$ & Index of activation map\\
            $M$ & Width of activation map\\
            $N$ & Height of activation map\\
            $S$ & Standalone model\\
            $\boldsymbol{\mu}$ & Mean\\
            $\boldsymbol{\sigma}$ & Standard deviation\\
            $\boldsymbol{x}$ & Image dataset
        \end{tabular}
    \end{minipage}
}

### Activation matching
<!-- 
- idea of activation matching is to find "similar" activation maps
- balancing problem of switching enough layers to get capture the characteristics of the standalone model, but limit the number of layers to be switched such that the the CLIP concept space embedding similarities remain consistent with what the text encoder learned. -->
Incorporating the properties of the standalone model to be explained into the CLIP image encoder is a delicate balancing act. On the one hand, we want to have all the standalone model's properties be explained and integrated into the CLIP image encoder to obtain the most significant explanation. On the other hand, the learned concept space of the CLIP embedding similarities needs to be maintained.

#TODO add image about balancing problem

To address this delicate balancing act, all activations of the standalone model are available for the selection process to be incorporated into the CLIP image encoder. To maintain the CLIP concept space as much as possible, only the last convolution layer of the last four out of the five available stages from the modified CLIP ResNet can be swapped. The last convolution layer of the first stage is skipped because we assume there is little information contained since earlier layers typically learn similar low-level concepts and the deeper layers behave very differently according to the task of the model.

#TODO add image about which activation of which model are changed (highlight in green)

Due to the imbalance in the number of available activation maps between the standalone model to be explained and the CLIP image encoder, there is a need for a suitable selection process. The name of this selection process is "Activation matching". The idea is to find activation maps in the standalone model which are "similar" to the activation maps in the CLIP image encoder. To evaluate which activations maps are similar, the activations of all convolution kernels of both models are normalized using a standard scaler and the previously computed statistics. (Compute mean and standard deviation of all activation maps of the dataset during inference of both models)

<!-- 
- normalize standalone activations
- normalize clip activations
-->
\noindent\fbox{
    \begin{minipage}{\linewidth}

        \begin{equation}
            \boldsymbol{N}^C_{k} = \frac{\boldsymbol{A}^C_{k} - \boldsymbol{\mu}^C_{k}}{\boldsymbol{\sigma}^C_{k}}
        \end{equation}

        \begin{equation}
            \boldsymbol{N}^S_{k} = \frac{\boldsymbol{A}^S_{k} - \boldsymbol{\mu}^S_{k}}{\boldsymbol{\sigma}^S_{k}}
        \end{equation}

        \begin{tabular}{l @{ $=$ } l}
            $\boldsymbol{A}$ & Activation maps\\
            $C$ & CLIP image encoder\\
            $\boldsymbol{N}$ & Normalized activation maps\\
            $k$ & Index of activation map\\
            $S$ & Standalone model\\
            $\boldsymbol{\mu}$ & Mean\\
            $\boldsymbol{\sigma}$ & Standard deviation\\
        \end{tabular}
    \end{minipage}
}

<!-- 
- upscale activation maps using bilinear interpolation
- upscale activation maps using bilinear interpolation -->
Since the activation maps to be compared could be of different sizes, the smaller one of the two activation maps is upscaled using a bilinear transformation to match sizes.

#TODO add image of bilinear transformation

<!-- 
- compute scores
scores = torch.einsum('aixy,ajxy->ij', standalone_model_activation_scaled, clip_model_activation_scaled)/(batch_size*map_size**2)   -->
These upscaled activation maps are used to find the most similar activation maps between the two models. This process is called "activation matching".

\noindent\fbox{
    \begin{minipage}{\linewidth}

        \begin{equation}
            \boldsymbol{Z}_{ij} = \frac{\sum_{b=1}^{B} \sum_{w=1}^{W} \sum_{h=1}^{H} \boldsymbol{N}^S_{biwh} \cdot \boldsymbol{N}^C_{bjwh}}{B \cdot W \cdot H}
        \end{equation}

        \begin{tabular}{l @{ $=$ } l}
            $B$ & Batchsize\\
            $C$ & CLIP model\\
            $H$ & Height of activation map\\
            $i$ & Activation map index of standalone model\\
            $j$ & Activation map index of CLIP image encoder\\
            $\boldsymbol{N}$ & Normalized activations\\
            $S$ & Standalone model\\
            $W$ & Width of activation map\\
            $\boldsymbol{Z}$ & Scores\\
        \end{tabular}
    \end{minipage}
}

The dimension of the scores matrix is $dim(\boldsymbol{s}_{ij}) = 22720 \times 3840$ filled with the valid range of values $\boldsymbol{s}_{ij} = [0, 1]$. The value $22720$ describes the number of convolutional kernels in the standalone model available for swapping. The value $3840$ describes the number of convolutional kernels in the CLIP image encoder available for swapping. Each score describes how "similar" the scaled activation maps of the standalone model and the CLIP image encoder are relative to each other. The similarity measurement is a trivial sum of products to limit the computing power needed. Therefore, a large score results from two large factors. A small score results from at least one small factor in the product. Ambiguous scores around 0.5 could occur for a small factor and a large one, two medium-sized factors or a large one and a small one.

### Swapping layers
<!-- 
- incorporate standalone into clip
- switch 3840 of 22720 from standalone to clip
- rescale activation maps accordingly
- formula inverse std scaler
- image layer swapping
    - first challenge -> Different  kernel sizes -> Upscale
    - second challenge -> Different scales -> apply inverse standard scaler
-->
#TODO maybe add an image
Scanning the score matrix from the activation matching process for the top 3840 (Number of activations in CLIP image encoder to be swapped) out of 22720 scores (Available activation maps from the standalone model) results in a scheme which activation maps need to be swapped. Swapping two activation maps brings two challenges. First, the activation maps could have different sizes. Therefore, the activation map from the standalone model gets rescaled to the size of the original activation map from the CLIP image encoder using a bilinear transformation. The second challenge is to address the different scales of the activation maps. As explained in \*@sec:activation-matching, all activation maps have been scaled using a standard scaler, therefore they are mean free and have a variance equal to one. After an activation map has been swapped from the standalone model to the CLIP image encoder, the activation needs to be adjusted to the original CLIP scale using an inverse standard scaler and the original CLIP statistics.

\noindent\fbox{
    \begin{minipage}{\linewidth}

        \begin{equation}
            \boldsymbol{A}^X_{l} = \boldsymbol{N}^S_{k} \cdot \boldsymbol{\sigma}^C_{l} + \boldsymbol{\mu}^C_{l}
        \end{equation}

        \begin{tabular}{l @{ $=$ } l}
            $\boldsymbol{A}$ & Activation maps\\
            $C$ & CLIP image encoder\\
            $k$ & Matching standalone activation map index according to scores $\boldsymbol{Z}$\\
            $l$ & Matching CLIP activation map index according to scores $\boldsymbol{Z}$\\
            $\boldsymbol{N}$ & Normalized activation maps\\
            $S$ & Standalone model\\
            $X$ & Caption-based explainable AI model\\
            $\boldsymbol{\mu}$ & Mean\\
            $\boldsymbol{\sigma}$ & Standard deviation\\
        \end{tabular}
    \end{minipage}
}

## Inference
After the network surgery, the caption-based explainable AI model consists of CLIP's original text encoder (Purple) and the modified image encoder (Red/Green striped) as shown in \*@fig:network_surgery_result_unbiased. The hypothesis is that the network surgery process can merge decision-critical properties and preserve the CLIP embedding space at the same time. 

![The caption-based explainable AI model consists of the original CLIP text encoder (Purple) and the post-network surgery image encoder (Red/Green striped). The highest score of the embedding similarities indicates which caption describes the image the best from the original standalone model (Red) point of view.](source/figures/network_surgery_result_unbiased.png "Post-network surgery caption-based explainable AI model."){#fig:network_surgery_result_unbiased width=100%}

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

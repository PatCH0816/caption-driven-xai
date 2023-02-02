# Abstract {.unnumbered}
A fundamental property of machine learning (ML) models is that they are not explicitly programmed but learn from data instead. This attribute makes them very powerful but challenging to interpret. The science of interpreting ML models to understand their behavior and improve their robustness is called explainable artificial intelligence (XAI). One of the state-of-the-art XAI methods for computer vision problems is to generate beneficial saliency maps. A saliency map highlights the pixel space on which the model focuses the most. However, this property could be misleading if decision-critical and correlating features are present in overlapping pixel spaces.

Introducing a modified version of the MNIST dataset with a color-encoded covariate shift between the datasets available during the development and a simulated real-world situation allows us to demonstrate the novel caption-based explainable AI method. The covariate shift is caused by a highly correlating feature (Color of the digit) in the same pixel space as the decision-critical feature (Shape of the digit). An ML model learns the more accessible correlating color feature instead of the decision-critical shape feature. In a real-world situation, the color assignments of the digits are random. Therefore, the ML model fails in a real-world situation. A saliency XAI approach would fail to explain this troubling situation since it would highlight the colored digit. A novel network surgery approach fuses the ML model to be explained with the contrastive language-image pre-training (CLIP) model. The resulting caption-based explainable AI model uses a set of captions to find the most descriptive text for a given image. This property enables the caption-based explainable AI method to express which concept the ML model focuses on instead of which pixel space.

The resulting caption-based explainable AI model can identify the dominant concept that contributes the most to the model's predictions. The most promising result is the superiority of the novel XAI method over saliency maps in specific situations. The central thesis validated by this work is that a deeper understanding of the dominant concepts in convolutional neural networks is fundamental and can ultimately improve the model's robustness. Our findings suggest that this novel XAI method should not just be seen as a pure debugging tool but as a necessary prerequisite before deploying any machine vision convolutional neural network model.





<!-- \hypertarget{abstract}{%
\chapter*{\vspace*{-4cm} Abstract}\label{abstract}}
\addcontentsline{toc}{chapter}{Abstract} -->

<!-- Short version of introduction and results -->
<!-- Initial situation: (3/ 9 sentences)
- Machine learning in all areas of our lives.
- Safety/Need for robustness
- Explainable AI -->
<!-- 
Babytalk:
After a few short years of life, children can fathom the concepts behind simple words and connect them to related images. They can identify the connection between shapes and textures of the physical world to the abstract symbols of written language. It’s something we take for granted. Very few (if any) people in the world will remember a time when these “basic” skills were beyond their capacity.
-->

<!-- 
Very nice intro inspiration:
In recent years, there’s been an explosion of AI datasets and models that are impacting millions around the world each day. Some systems are recommending us songs and movies to enjoy; others are automating fundamental business processes or saving lives by detecting tumors. In the near future, machines relying on AI will drive our cars, fly our aircraft, and manage our care. But for that to take place, we need to ensure that those systems are robust enough against any attempts to hack them.
asdf
During development of an AI model, conditions are carefully controlled to obtain the best possible performance — like starting a seedling in a greenhouse. But in the real-world, where models are ultimately deployed, conditions are rarely perfect, and risks are abundant. If development is a greenhouse, then deployment is a jungle. We have to prepare AI models to withstand the onslaught.
asdf
For years, AI models struggled to reach accuracy levels suitable for real-world applications. Now that they’ve reached that threshold for certain tasks, it is crucial to recognize that accuracy isn’t the only benchmark that matters. In the real-world, fairness, interpretability, and robustness are critical, and many tools are available to inspect these dimensions of AI models. Developers must actively prepare AI models to succeed in the wild by spotting holes in the armor, predicting an adversary’s next move, and weaving robustness into the fabric of AI.
-->
<!-- 
An increasing number of new and exciting machine learning applications disrupt our lives almost daily. Autonomous driving, movie recommender systems and traffic prediction are just a few large-scale projects to be mentioned in this context. There is no doubt that many more exciting applications will come, but with great power comes great responsibility. The fundamental property of machine learning models is that they are not explicitly programmed but learn from data instead. This characteristic makes them very powerful but challenging to interpret. The situation gets even worse in e.g. medical environments where a patient could suffer from wrong predictions made by a machine learning model. Therefore, it is mission-critical to deploy robust machine learning models only. The science of interpreting machine learning models to improve their robustness is called explainable artificial intelligence (XAI). One of the state-of-the-art methods in computer vision problems is generating beneficial saliency maps. The primary issue of these saliency maps is that they tell where the model focuses on an image instead of answering what they are seeing. This property is problematic if correlating and causating features are present in overlapping pixel space.  -->

<!--
Approach/Technology: (10/12 sentences)
- XAI helps to understand, does not fall for a bias (correlation instead of causation)
- state of the art saliency maps tell where, but not why or what
- increase robustness
- Dataset with covariate shift.
- biased model falls for bias instead of learning the task.
- Accuracy looks promising -> Is dangerous!
- Using CLIP to explain
- New XAI method: Network surgery/Layer swapping
- Hypothesis: CLIP can be used to obtain an explanation -> Points to bias and covariate shift -> Covariate shift is understood and can be fixed

Introducing a modified version of the MNIST dataset with a covariate shift between the training/validation and test subsets allows us to demonstrate the novel XAI method. The covariate shift is caused by a positive correlating feature (Color) in the same pixel space as the causation feature (Shape of the digit) for the training and validation subsets. A negative correlating feature (Color) in the same pixel space as the causation feature (Shape of the digit) for the test subset. A standalone model most likely learns the more accessible to learn correlating feature (Color) instead of the causating (Shape of the digit) feature. Therefore, the model fails miserably on the test dataset. A saliency XAI approach would fail to explain this troubling situation since it would highlight the colored digit. The novel XAI method presented uses a contrastive language-image pre-training (CLIP) model to obtain an explanation in natural language. This core characteristic enables the XAI method to express what the model focuses on instead of where. A novel network surgery approach switches layers between a given image classification model to be explained and the image encoder of CLIP. During this layer-switching process, the properties of the embedding space of CLIP and the decision-critical layers from the model to be explained are merged and preserved. The novel XAI method enables AI practitioners to ask questions like "Do you see a red digit?" (Undesired correlating feature) or "Do you see a digit with the number 5?" (Desired causation feature) and to understand and improve the model's behavior by interpreting the results. -->

<!-- Result/Conclusion: (6/2 sentences)
The resulting explanation provides an opportunity to improve the model on possible concerning findings. Therefore, the final product is a trusted machine learning model which uses robust features instead of e.g. spurious correlations. Work in progress...
-->

<!-- Total: (19/23 sentences) - maximum: 3500 characters -->

\pagenumbering{roman}
\setcounter{page}{1}
\newpage

<!-- Nice sentences: -->
<!-- Why should we trust AI enough to drive cars, detect diseases, and identify suspects when it is a black box? -->

<!-- 
#TODO 
[x] Text Amil for a meeting
[x] Explain network dissection
[x] Interpretability vs. explainability
[x] check if image encoder from clip and openclip look the same
    [x] Show differences between CLIP and open-CLIP in appendix
[x] Replace with better definition of contrastive learning!
[x] Clean desktop
[x] Add CLIP limitations/advantages/disadvantages
[x] Understand einsum(..) -> scores = torch.einsum('aixy,ajxy->ij', ... -> torch.product(..) and toch.sum(..)
[x] check 1024 features are dependent if resnet or transformer
[x] update dataloader to provide CLIP preprocessored images
[x] whats the scale of the original mnist data? 0-1 or 0-255? -> CLIP expect 0-1
[x] balance 5/8 mnist dataset
[x] explain numeric differences between CLIP and open-CLIP
[x] fix presentation date for lab buddies (08.02@lab-meeting and 09.02 presentation at 08:00 o'clock)
[x] Chapter model add learning curves
[x] gray biased mnist to gray unbiased mnist
[x] introduce term "embedding similarities" (clip.png)
[x] change human robot image with captions instead of suggestions!
[x] chapter model: replace 2 learning curves figures and add accuracy on real-world dataset (~0%)
[x] Show differences between ResNet-50 and CLIP's modified ResNet-50 in appendix
[x] draw images in abstract
[x] add attended courses and seminars to closing words
[x] chapter results: add unbiased/gray learning curves figures and add accuracy on real-world dataset (~100%)
[x] retrain models with random color assignments
[x] coding
    - preparation
        [x] setup dataset
        [x] prepare standalone model
        [x] prepare clip model
        [x] compute mean+std for each layer
    - matching
        [x] match layers
        [x] plot matching scores
    - layer swapping
        [x] swap layers
        [x] observe cosine similarities from clip/analyze impact of network surgery
    - Order of tasks:
        [x] Understand content in standalone_statistics
        [x] Understand content in clip_statistics
        [x] Understand einsum(..)
        [x] Understand content in table
        [x] Understand match_scores
        [x] Understand swapping
    [x] Add another representation with cosine similarities and probabilities in layer swapping plots
    [x] Transfer learning: Unfreeze all layers and retrain models
    [x] Extend ResNet explanation
[x] references: format for double quotes like ""asdf"", etc.
[x] references: publisher is not displayed! Replace with journal if applicable!
[x] document hyperparameters for standalone model training
[x] document size of dataset splits
[x] Table of contents is missing
[x] List of figures is missing
[x] document network surgery
[x] check for correct usage: convolution kernel/convolutional layer/convolutional neural network/activation map
[x] Mention configuration for images in chapter "results"
[x] describe tables in appendix
[x] move shape/color figures to appendix  
[x] leave test figures in results and move others to the appendix
[x] evtl. create new figures showing all four values and put them in front of the bundled figure
[x] document results
[x] correct introduction
[x] write abstract for documentation (limit to 1 page) and online
[] document conclusion

# Bonus:
[] Why are cosine similarities always larger than 0? Because of the log from the loss function used for training?
[] Preprocessor size of images in chapter dataset
-->

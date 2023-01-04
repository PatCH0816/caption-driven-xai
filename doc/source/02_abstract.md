# Abstract {.unnumbered}

<!-- \hypertarget{abstract}{%
\chapter*{\vspace*{-4cm} Abstract}\label{abstract}}
\addcontentsline{toc}{chapter}{Abstract} -->

<!-- Short version of introduction and results -->
<!-- Initial situation: (3/ 9 sentences)
- Machine learning in all areas of our lives.
- Safety/Need for robustness
- Explainable AI -->
An increasing number of new and exciting machine learning applications disrupt our lives almost daily. Autonomous driving, movie recommender systems and traffic prediction are just a few large-scale projects to be mentioned in this context. There is no doubt that many more exciting applications will come, but with great power comes great responsibility. The fundamental property of machine learning models is that they are not explicitly programmed but learn from data instead. This characteristic makes them very powerful but challenging to interpret. The situation gets even worse in e.g. medical environments where a patient could suffer from wrong predictions made by a machine learning model. Therefore, it is mission-critical to deploy robust machine-learning models only. The science of interpreting machine learning models to improve their robustness is called explainable artificial intelligence (XAI). One of the state-of-the-art methods in computer vision problems is generating beneficial saliency maps. The primary issue of these saliency maps is that they tell where the model focuses on an image instead of answering what they are seeing. This property is problematic if correlating and causating features are present in overlapping pixel space. 

<!-- Approach/Technology: (10/12 sentences)
- XAI helps to understand, does not fall for a bias (correlation instead of causation)
- state of the art saliency maps tell where, but not why or what
- increase robustness
- Dataset with covariate shift.
- Fooled model falls for bias instead of learning the task.
- Accuracy looks promising -> Is dangerous!
- Using CLIP to explain
- New XAI method: Network surgery/Layer swapping
- Hypothesis: CLIP can be used to obtain an explanation -> Points to bias and covariate shift -> Covariate shift is understood and can be fixed -->
Introducing a modified version of the MNIST dataset with a covariate shift between the training/validation and test subsets allows us to demonstrate the novel XAI method. The covariate shift is caused by a positive correlating feature (Color) in the same pixel space as the causation feature (Shape of the digit) for the training and validation subsets. A negative correlating feature (Color) in the same pixel space as the causation feature (Shape of the digit) for the test subset. A standalone model most likely learns the more accessible to learn correlating feature (Color) instead of the causating (Shape of the digit) feature. Therefore, the model fails miserably on the test dataset. A saliency XAI approach would fail to explain this troubling situation since it would highlight the colored digit. The novel XAI method presented uses a contrastive-language-image-pre-training (CLIP) model to obtain an explanation in natural language. This core characteristic enables the XAI method to express what the model focuses on instead of where. A novel network surgery approach switches layers between a given image classification model to be explained and the image encoder of CLIP. During this layer-switching process, the properties of the embedding space of CLIP and the decision-critical layers from the model to be explained are merged and preserved. The novel XAI method enables AI practitioners to ask questions like "Do you see a red digit?" (Undesired correlating feature) or "Do you see a digit with the number 5?" (Desired causation feature) and to understand and improve the model's behavior by interpreting the results.

<!-- Result/Conclusion: (6/2 sentences) -->
The resulting explanation provides an opportunity to improve the model on possible concerning findings. Therefore, the final product is a trusted machine-learning model which uses robust features instead of e.g. spurious correlations. Work in progress...

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
[] Interpretability vs. explainability
[] check if image encoder from clip and openclip look the same
[] Show differences between CLIP and open-CLIP in appendix
[] Clean desktop
[] Draw network surgery
[] Understand einsum(..)
[] Coding.. :P
-->

# Abstract {.unnumbered}

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

During development of an AI model, conditions are carefully controlled to obtain the best possible performance — like starting a seedling in a greenhouse. But in the real world, where models are ultimately deployed, conditions are rarely perfect, and risks are abundant. If development is a greenhouse, then deployment is a jungle. We have to prepare AI models to withstand the onslaught.

For years, AI models struggled to reach accuracy levels suitable for real-world applications. Now that they’ve reached that threshold for certain tasks, it is crucial to recognize that accuracy isn’t the only benchmark that matters. In the real world, fairness, interpretability, and robustness are critical, and many tools are available to inspect these dimensions of AI models. Developers must actively prepare AI models to succeed in the wild by spotting holes in the armor, predicting an adversary’s next move, and weaving robustness into the fabric of AI.
-->

An increasing number of new and exciting machine learning applications disrupt our lives almost daily. Autonomous driving, movie recommender systems and traffic prediction are just a few large-scale projects to be mentioned in this context. There is no doubt that many more exciting applications will come, but with great power comes great responsibility. The fundamental property of machine learning models is that they are not explicitly programmed but learn from data instead. This characteristic makes them very powerful but challenging to interpret. The situation gets even worse in e.g. medical environments where a patient could suffer from wrong predictions made by a machine learning model. Therefore, it is mission-critical to deploy robust machine-learning models only. The science of interpreting machine learning models to improve their robustness is called explainable artificial intelligence (XAI). One of the state-of-the-art methods in computer vision problems is generating beneficial saliency maps. The primary issue of these saliency maps is that they tell where the model focuses on an image instead of answering what they are seeing. This property is problematic if correlating and causating features are present in overlapping pixel space. 

<!--
Approach/Technology: (10/12 sentences)
- XAI helps to understand, does not fall for a bias (correlation instead of causation)
- state of the art saliency maps tell where, but not why or what
- increase robustness
- Dataset with covariate shift.
- Fooled model falls for bias instead of learning the task.
- Accuracy looks promising -> Is dangerous!
- Using CLIP to explain
- New XAI method: Network surgery/Layer swapping
- Hypothesis: CLIP can be used to obtain an explanation -> Points to bias and covariate shift -> Covariate shift is understood and can be fixed
-->
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
[x] Interpretability vs. explainability
[x] check if image encoder from clip and openclip look the same
    [x] Show differences between CLIP and open-CLIP in appendix
    [] Show differences between ResNet-50 and CLIP's modified ResNet-50 in appendix
[x] Replace with better definition of contrastive learning!
[x] Clean desktop
[x] Add CLIP limitations/advantages/disadvantages
[x] Understand einsum(..) -> scores = torch.einsum('aixy,ajxy->ij', ... -> torch.product(..) and toch.sum(..)
[x] check 1024 features are dependent if resnet or transformer
[] dataloader provides imgs not ready for clip (preprocess images in dataloader)
[] balance 5/8 mnist dataset
[] network surgery
[] fix presentation date for lab buddies (09.02 presentation at 09:00 o'clock)
[] Draw network surgery
-->

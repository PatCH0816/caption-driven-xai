# Explainable AI
<!--- What is an explanation method? Why is it needed? -->
<!-- Why Care About Interpretability?
1. Help building trust:
• Humans are reluctant to use ML for critical tasks
• Fear of unknown when people confront new technologies
2. Promote safety:
• Explain model’s representation (i.e. important feature)
providing opportunities to remedy the situation
3. Allow for contestability:
• Black-box models don't decompose the decision into submodels or illustrate a chain of reasoning -->
<!--- Different state of the art approaches -->
It is critical to unearth hidden problems in real-world data science and not to fall for "correlation is not causation" and other types of problems. Accepting the fact that these challenges exist is the first step to improvement. At first, one needs to understand what a machine learning model is doing. The right tool for that kind of task are methods from the explainable artificial intelligence (XAI) toolbox. XAI tools help to promote safety, allow for contestability and help build trust in models. This chapter provides an overview of the difference between interpretability and explainability, the problem with covariate shifts in data distributions, state-of-the-art XAI saliency maps for machine-vision problems and a recent XAI method called network dissection.

## Interpretability vs. explainability
Even tho the terms "interpretability" and "explainability" are used interchangeably in XAI literature, there is a significant difference between them. Considering an example of boiling water, the data would show a continuously rising water temperature until ~100°C. Around this temperature, the water temperature would remain stable. A machine learning (ML) model can learn this behavior and make reliable predictions without knowing the physics of the changing state of water from liquid to steam. According to Gianfagna et al: "..we will consider interpretability as the possibility of understanding the mechanics of a Machine Learning model but not necessarily knowing why." [@xai_gianfagna_dicecco] Explainability, on the other hand, would require an extended ML model, which is aware of the changing state of water and its related physics phenomena. This water analogy leads to the definition of Gilpin et al: "..explainable models are interpretable by default, but the reverse is not always true." [@gilpin_explainability_interpretability]

## Covariate shift
<!-- There are three different types of dataset shifts:

https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/
http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf
- Shift in the independent variables (Covariate Shift)
- Shift in the target variable (Prior probability shift)
- Shift in the relationship between the independent and the target variable (Concept Shift)

All three mentioned shifts could have a negative impact on the performance of a machine learning model, but this work focuses solely on the covariate shift. -->
XAI tools help discover non-obvious problems with the machine learning model. One dangerous problem goes by the name "covariate shift". A covariate shift occurs when the distribution of the independent variables in the training dataset differs from the test dataset. [@covariante_shift] \*@fig:covariate_shift_regression illustrates the challenge if the training samples do not represent the test samples accordingly in a regression problem. This situation could either arise because of inadequate data acquisition or a terrible choice of train/test splits.

![This illustration demonstrates the negative impact of the covariate shift on the success of the machine learning model trying to learn a true function (Red curve). Given the training samples (Blue dots), the model learns the linear learned function (Green line). The performance on the test samples (Black) will be terrible because the learned function does not approximate the true function very well in the space around the test samples. In an ideal setting, the training samples should have been equally distanced and scattered over the whole space of the true function with as low variance as possible. [[@covariate_shift_regression]](#references)](source/figures/covariate_shift_regression.png "Covariate shift demonstration for a regression problem"){#fig:covariate_shift_regression width=50%}

The danger of facing a covariate shift in data distributions is a general problem. This phenomenon occurs in regression problems, natural language processing, computer vision, and other data representations. The universal language model BERT, which Google has developed, can understand sentences and generate suitable embeddings. Creating such a powerful model requires a massive amount of data, which inevitably contains embedded biases. For example, a specific name always has a negative connotation or certain words, which are associated with one gender over the other, independent of the context. [@bert_bias]

A real-life covariate shift and confirmation bias can be observed in schools regularly. Students who study for an upcoming examination solving similar exams from earlier years without gaining general knowledge about a topic, will increase their confirmation bias in their abilities. If the teacher decides to completely change the type of exercises in the upcoming exam (Covariate shift between the type of exercises in earlier exams and the upcoming exam), these types of students will suffer. Analogous machine learning models optimized for a benchmark can achieve a high train/validation accuracy and a similar or slightly lower test accuracy. This accuracy can be interpreted as an upper bound for the expected accuracy on another dataset from the same distribution. If the dataset suffers from a covariate shift, the accuracy could drop arbitrarily low due to the lack of a lower bound for the expected accuracy on the new domain.

In a final example, which takes place in the context of a hospital, a patient could suffer dangerous consequences if such a covariate shift in a deployed machine learning model remains undetected. A team of artificial intelligence (AI) researchers and radiologists claims to be able to detect coronavirus disease of 2019 (COVID-19) artefacts from chest radiographs with their new machine learning model. However, experiments reveal that high accuracy is not achieved because of actual medical pathology features but because of confounding factors. In the worst possible scenario, a different hospital provides data with similar confounding factors since they use the same type of x-ray machine or other factors. These findings lead to an alarming situation where the machine learning model appears accurate but fails when tested in other hospitals. [@covid_shortcuts_over_signal] Using accuracy as the only metric to measure the performance of a model is dangerous. The metric accuracy could suffer from confirmation bias as this COVID-19 example demonstrates.

## Saliency maps
This work focuses on machine vision problems. A widely used XAI method is generating saliency maps to understand which image region excites the machine learning model the most for a specific class. Saliency maps highlight an area of pixels that contribute the most to the actual prediction. [@saliency_maps] \*@fig:wolves_and_dogs_prediction demonstrates where saliency maps are helpful. The task is to classify the images into wolves and huskies. The results from \*@fig:wolves_and_dogs_prediction indicate an accuracy of the model of about $\frac{5}{6} \approx 83\%$ to classify huskies and wolves.

![Shown is a binary classification task on six images of wolves and dogs. Five out of six predictions are correct. [[@wolves_and_dogs_prediction]](#references)](source/figures/wolf_or_husky.png "Wolf or husky predictions"){#fig:wolves_and_dogs_prediction width=100%}

Using a saliency map with a threshold as in \*@fig:husky_saliency_map indicates that the machine learning model did not focus on expected features like the fur's colors, the ear's shape or the snout's length to distinguish between wolves and huskies, but on the background. Therefore, the model just learned to distinguish between "snow" and "no snow" in the background and failed to learn the actual task due to spurious correlation. [@xai_gianfagna_dicecco]

![Image (a) shows a husky, classified as a wolf. The saliency map in the image (b) provides a visual explanation that the model ignored the animal and focused on the snow in the background instead. [[@wolves_and_dogs_xai]](#references)](source/figures/husky_saliency_map.png "Husky classified as wolf."){#fig:husky_saliency_map width=80%}

\*@fig:husky_saliency_map demonstrates successfully how powerful XAI methods identify problems with a model under test. On top of that, XAI can also increase humans' trust in machine learning models. 27 graduate students who have taken at least one graduate machine learning course were asked in a survey if they trust the model to classify huskies and wolves. The survey kept track if they mentioned the snow as a potential feature. As soon as the student answered the question, the explanation image with the highlighted snow in the background in \*@fig:husky_saliency_map was revealed. Then, the same question was asked again. The findings are collected and displayed in the \*@tbl:survey_table. Without the saliency map, about half of the students had some faith in the flawed model and about half of the students thought the snow could be a potential feature. After revealing the explanation with the saliency map, only three out of 27 students still maintained their trust in the flawed model. 25 out of 27 students mentioned that the snow in the background is a potential feature. This survey shows that XAI methods can leverage the level of trust of humans into machine learning models. [@wolves_and_dogs_xai]

|                               | Before        | After
|-                              | -             | -
|Trusted the bad model          | 10 out of 27  | 3 out of 27
|Snow as a potential feature    | 12 out of 27  | 25 out of 27
Table: Some graduate students who have taken at least one graduate machine learning course were asked if they trust the model to classify huskies and wolves and if they mentioned the snow as a potential feature. The same questions were asked before and after revealing the explanation image with the highlighted snow in the background. [@wolves_and_dogs_xai](#references) {#tbl:survey_table}

<!-- https://www.statworx.com/content-hub/blog/car-model-classification-3-erklarbarkeit-von-deep-learning-modellen-mit-grad-cam/#:~:text=Grad%2DCAM%20erweitert%20die%20Anwendbarkeit,jede%20der%20entsprechenden%20Feature%20Maps. -->
One of the most widespread XAI methods which produce saliency maps is the gradient class-activation-map (Grad-CAM) method. The Grad-CAM method uses the gradients of the classification score to weight the activation maps, which ultimately highlight the pixels in the input image with the most significant influence on the classification score. [@grad_cam] This Grad-CAM method is considered incredibly useful for comprehending what the machine learning model is doing. However, a common disadvantage in all saliency map-producing methods is that they only highlight pixel spaces of an image that contribute the most to a prediction. Saliency methods do not reveal which concept the ML models believe in seeing in that pixel region of the image.

<!--
http://netdissect.csail.mit.edu/ (Paper)
https://paperswithcode.com/method/network-dissection
https://medium.com/analytics-vidhya/demystifying-hidden-units-in-neural-networks-through-network-dissection-7d3ac657c428 (Simplified explanation of the paper in form of a blog)
-->
## Network dissection
<!-- 
- Researcher at MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)
- What is going on inside of a neural network?
- Discover concepts learned by internal activations of models
- BroDen dataset is a diverse dataset. 
- Idea to keep track of all activations of each neuron to each of the images
- In other words, this method interprets networks by providing meaningful labels to their hidden units.
- In the past, observations of hidden units have shown that human-interpretable concepts sometimes emerge in individual units within networks.
- Human-interpretable concepts include low-level concepts like colors and high-level concepts such as objects. By measuring the concept that best matches each unit, Net Dissection can break down the types of concepts represented in a layer. -->

<!-- Introduction -->
Deep neural networks are complex ML models. The wider audience depicts deep neural networks as being black boxes. It would be fascinating to cut such networks open to analyze and understand the task of each hidden unit. A reasonably recent method called "Network dissection" has been introduced by a team of researchers from the MIT Computer Science and Artificial Intelligence Laboratory (CSAIL), which enables one to comprehend what kind of concepts a hidden unit is looking for. [@network_dissection] 

<!-- Spoiler of how it works -->
The basic idea is to stimulate the network with a broad range of visual concepts captured in images to discover concepts learned in the internal activations a model. In the end, network dissection can tell which layer captures which concept(s). Since the matching algorithm of the new XAI method is an adaptation of this method, the tripartite process is explained in the following section to get a gentle introduction to the presented idea without going too much into the details.

\noindent
**Dataset**  
The authors created a new broadly and densely labeled (BroDen) dataset, which contains about 60'000 images of textures, object parts, materials and scenes. The BroDen dataset unites many existing densely labeled image datasets like Pascal-Parts [@pascal_parts_dataset], Pascal-Context [@pascal_context_dataset], ADE [@ade_dataset], describable textures [@describable_textures_dataset] and Open Surfaces [@open_surfaces_dataset]. The essential property of this dataset is its size and the fact that all images are labeled pixelwise. This means there is a binary saliency map for each image, which tells where the metal pan, the dog or the sand are located in an image. Therefore, there is a binary segmentation map $\boldsymbol{L}_c$ for all concepts $\boldsymbol{c}$.

\noindent
**Track activation maps**
Images from the BroDen dataset are fed into the neural network to be explained. All activations of all hidden units are tracked during this process to match the responses to their corresponding concepts. In more detail, the process looks like this:

- Each image $\boldsymbol{x}$ of the BroDen dataset is fed into the neural network to be explained. All activation maps $\boldsymbol{A}_k$ for each unit $\boldsymbol{k}$ are stored during this inference process.
- Use $\boldsymbol{A}_k$ over all images $\boldsymbol{x}$ to compute the distribution of activations $\boldsymbol{a}_k$.
- To compare the binary segmentation maps from the BroDen dataset with the distribution of activations $\boldsymbol{a}_k$ for a given unit $k$, $\boldsymbol{a}_k$ ultimately needs to be converted to a binary map. Compute a threshold $\boldsymbol{T}_k$, such that 0.5% of all activations of unit $k$ are greater than $\boldsymbol{T}_k$, more mathematically: $P(\boldsymbol{a}_k > \boldsymbol{T}_k) = 0.005$. 
- The deeper the activation maps in a neural network, the smaller its size. Applying a bilinear interpolation to all lower-scale activation maps $\boldsymbol{A}_k(\boldsymbol{x})$ allows scaling of all activation maps to the same size as the original input image. The scaled activation maps are called $\boldsymbol{S}_k(\boldsymbol{x})$.
- The binarized activation maps $\boldsymbol{M}_k(\boldsymbol{x})$ are the result of applying the threshold $\boldsymbol{T}_k$ to the scaled activation maps $\boldsymbol{S}_k(\boldsymbol{x})$ like this: $\boldsymbol{M}_k(\boldsymbol{x}) = \boldsymbol{S}_k(\boldsymbol{x}) >= \boldsymbol{T}_k(\boldsymbol{x})$. This activation masks $\boldsymbol{M}_k(\boldsymbol{x})$ highlights the pixels space, which is responsible for maximizing the activation for a unit $k$ given an image $\boldsymbol{x}$.

\noindent
**Align activation maps with concepts**
In the final step, the activation maps and the concepts are aligned to understand which activation maps correspond to which concepts. To find the concepts each node is looking for, the activation masks $\boldsymbol{M}_k$ and the human-labeled concept masks $\boldsymbol{L}_c$ are compared. The intersection over union (IoU) score is used as a measure of similarity.

![A human immediately recognizes a body underneath the scarf, which belongs to the pug. (Human annotated ground truth) Most machine vision models do not know about partially covered objects. Therefore only the face of the pug is detected. (Top activated area) These two quantities allow us to compute the intersection area and union area, which are ultimately needed to compute the intersection over union. [[@intersection_over_union]](#references)](source/figures/intersection_over_union.jpg "Intersection over union"){#fig:intersection_over_union width=100%}

The IoU score computes the number of overlapping pixels in both the concept mask and the activation mask divided by the total number of unique pixels in concept $\boldsymbol{c}$.

\noindent\fbox{
    \begin{minipage}{\linewidth}
        \begin{equation}
            IoU_{k,c} = \frac{\sum | M_k(\boldsymbol{x}) \cap L_c(\boldsymbol{x}) |}      {\sum | M_k(\boldsymbol{x}) \cup  L_c(\boldsymbol{x}) |} 
        \end{equation}
        \begin{tabular}{l @{ $=$ } l}
            $\boldsymbol{x}$ & Input image\\
            $k$ & Hidden unit\\
            $c$ & Concept\\
            $M_k$ & Activation mask for unit k\\
            $L_c$ & Human labelled image segment of concept c\\
            $IoU_{k,c}$ & Intersection over union score for unit k and concept c
        \end{tabular}
    \end{minipage}
}

This IoU score represents if a unit $k$ gets excited about a concept $\boldsymbol{c}$. If the IoU score exceeds a threshold to be defined, the neuron is considered a detector of concept $\boldsymbol{c}$. It is important to note that one unit could detect several concepts. The final findings to be highlighted in this paper are:

- The architecture of the neural network influences its interpretability: ResNet > VGG > GoogLeNet > AlexNet (Larger means better interpretability)
- Self-supervised neural networks have fewer unique concept detectors than those trained on supervised tasks.
- The first few layers (Close to the input) contain high-level concepts (E.g. texture and color) and deeper layers contain low-level features (E.g. parts and objects).
- Increasing the number of interpretable units in a layer increases the number of unique concept detectors. Adding batch normalization decreases the number of unique concept detectors.
- An increasing number of training iterations lead to a more significant number of unique concept detectors.

## Theory summary
Getting back to the COVID-19 chest radiographs example, XAI supports ML models to reduce the risk of learning undesired "shortcuts" instead of medically relevant pathology features. XAI is mainly used to understand and improve machine learning models, which suffer from low performance. Instead, XAI should be considered a prerequisite for all machine learning models, especially for those with significant responsibility, e.g. in healthcare environments. An ounce of prevention is worth a pound of cure. Hopefully, the vast amount of XAI methods will leverage trust, fairness and robustness of future applications.

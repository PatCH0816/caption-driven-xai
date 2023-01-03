# Explainable artificial intelligence
<!--- What is an explanation method? Why is it needed? -->
<!-- Why Care About Interpretability?
5
1. Help building trust:
• Humans are reluctant to use ML for critical tasks
• Fear of unknown when people confront new technologies
2. Promote safety:
• Explain model’s representation (i.e. important feature)
providing opportunities to remedy the situation
3. Allow for contestability:
• Black-box models don't decompose the decision into submodels or illustrate a chain of reasoning -->
It is mission-critical to unearth hidden problems in real-world data science and not to fall for "correlation is not causation" problems. Accepting the fact that these challenges exist is the first step to improvement. At first, one needs to understand what a machine learning model is doing. The right tool for that kind of task are methods from the explainable AI toolbox. Getting back to the COVID-19 chest radiographs example, XAI supports AI systems to reduce the risk of learning undesired "shortcuts" instead of medically relevant pathology features. XAI is mainly used to understand and improve machine learning models, which suffer from low performance. Instead, XAI should be considered a prerequisite for all machine-learning models with significant responsibility, e.g. in healthcare environments. Hopefully, the vast amount of XAI methods will enable "Data Justice", "Data Trust" and "Data Fairness" in future applications.

## Saliency maps
<!--- Different state of the art approaches -->
<!--- grad-cam heatmaps [@xai_gianfagna_dicecco] -->
Part of this thesis is to work on computer vision problems. A widely used XAI method is generating saliency maps to understand which image region excites the machine-learning model the most for a specific class. Saliency maps highlight an area of pixels that contribute the most to the actual prediction. [@saliency_maps]

\*@fig:wolves_and_dogs_prediction demonstrates where saliency maps are helpful. The task is to classify the images into wolves and huskies. Five out of six predictions are correct. One question remains: Is this a good classifier?

![Shown is a binary classification task on six images of wolves and dogs. Five out of six predictions are correct. [[@wolves_and_dogs_prediction]](#references)](source/figures/wolf_or_husky.png "Wolf or husky predictions"){#fig:wolves_and_dogs_prediction width=100%}

\*@fig:husky_saliency_map indicates that the machine-learning model did not focus on expected features like the fur's colors, the ear's shape or the snout's length to distinguish between wolves and huskies. The results from \*@fig:wolves_and_dogs_prediction suggest an accuracy of the model of about $\frac{5}{6} \approx 83\%$ to classify huskies and wolves. Therefore, the model just learned to distinguish between "snow" and "no snow" and failed to learn the actual task due to spurious correlation.

![Image (a) shows a husky, classified as a wolf. The saliency map in image (b) provides a visual explanation that the model ignored the animal and focused on the snow in the background instead. [[@wolves_and_dogs_xai]](#references)](source/figures/husky_saliency_map.png "Husky classified as wolf."){#fig:husky_saliency_map width=80%}

It has been demonstrated how powerful XAI methods reveal problems with a model under test. On top of that, XAI can also increase humans' trust in machine-learning models. 27 graduate students who have taken at least one graduate machine learning course were asked in a survey if they trust the model to classify huskies and wolves. The survey kept track if they mentioned the snow as a potential feature. As soon as the student answered the question, the explanation image with the highlighted snow in the background in \*@fig:husky_saliency_map was revealed. Then, the same question was asked again. The findings are collected and displayed in the \*@tbl:survey_table. Without the saliency map, about half of the students had some faith in the flawed model and about half of the students thought the snow could be a potential feature. After revealing the explanation with the saliency map, only three out of 27 students still maintained their trust in the flawed model. 25 out of 27 students mentioned that the snow in the background is a potential feature. This demonstration shows that XAI methods can leverage the level of trust of humans into machine-learning models.

|                               | Before        | After
|-                              | -             | -
|Trusted the bad model          | 10 out of 27  | 3 out of 27
|Snow as a potential feature    | 12 out of 27  | 25 out of 27
Table: Some graduate students who have taken at least one graduate machine learning course were asked if they trust the model to classify huskies and wolves and if they mentioned the snow as a potential feature. The same questions were asked before and after revealing the explanation image with the highlighted snow in the background. [@wolves_and_dogs_xai](#references) {#tbl:survey_table}

One of the most widespread XAI methods which produce saliency maps is the gradient class-activation-map (Grad-CAM) method. The Grad-CAM method uses the gradients of the classification score with respect to the final convolutional feature map to highlight the pixels in the input image with the most significant influence on the classification score. [@grad_cam] This Grad-CAM method is considered incredibly useful for comprehending what the machine learning model is doing. However, a common disadvantage in all saliency map-producing methods is that understanding where the machine-learning model focuses do not tell what it is doing with that region of interest. Therefore, saliency maps do not reveal what the model is thinking but where it is looking only.

## Interpretability vs. explainability
<!-- So What Is the Difference Between Interpretability and Explainability? -->
To provide a further visual example of this distinction between interpretability and 
explainability, let’s think about the boiling water; the temperature increases with 
time steadily until the boiling point after which it will stay stable. If you just rely on 
Fig. 1.12 Explanations decomposed (Deutsch 1998)
Fig. 1.13 An illustration of the error surface of Machine Learning model
1 The Landscape

data before the boiling point, the obvious prediction with the related interpretation 
would be that temperature rises continuously. Another interpretation may make 
sense of data taken after the boiling point with a steady temperature.
But if you search for a full explanation, a full theory of water “changing state,” 
this is something deeper that exceeds the single good interpretations and predic-
tions, in the two different regimes. ML would be good at predicting the linear trend 
and the fat temperature after the boiling point, but the physics of the phase transi-
tion would not be explainable (Fig. 1.14).

Interpretability would be to understand how the ML systems predict temperature 
with passing time in the normal regime; explainability would be to have a ML model 
that takes into account also the changing state that is a global understanding of the 
phenomenon more related to the application of knowledge discovery already 
mentioned.

To summarize, with the risk of an oversimplifcation of the discussion above but 
getting the core, we will consider interpretability as the possibility of understanding 
the mechanics of a Machine Learning model but not necessarily knowing why.

“We take the stance that interpretability alone 
is insuffcient. For humans to trust black-box methods, we need explainability – 
models that can summarise the reasons for neural network behaviour, gain the trust 
of users, or produce insights about the causes of their decisions. Explainable models 
are interpretable by default, but the reverse is not always true.”

## Network dissection
<!-- #TODO: Broden dataset
http://netdissect.csail.mit.edu/ (Paper)
https://paperswithcode.com/method/network-dissection
https://medium.com/analytics-vidhya/demystifying-hidden-units-in-neural-networks-through-network-dissection-7d3ac657c428 (Simplified explanation of the paper in form of a blog) -->
How to discover concepts learned in the internal activations of models. The idea is to apply the model to a very diverse dataset called Broden and keep track of the highest activations of each neuron to each of the images. [@network_dissection]

- Researcher at MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)
- What is going on inside of a neural network?
- Discover concepts learned by internal activations of models
- Broden dataset is a diverse dataset. 
- Idea to keep track of all activations of each neuron to each of the images
- In other words, this method interprets networks by providing meaningful labels to their hidden units.
- In the past, observations of hidden units have shown that human-interpretable concepts sometimes emerge in individual units within networks.
- Human-interpretable concepts include low-level concepts like colors and high-level concepts such as objects. By measuring the concept that best matches each unit, Net Dissection can break down the types of concepts represented in a layer.
1 The Broadly and Densely Labeled Dataset (Broden) unifies several densely labeled image data sets: ADE , Open Surfaces , Pascal-Context , Pascal-Part and Describable Textures Dataset. These data sets contain examples of a broad range of objects, scenes, object parts, textures, and materials in a variety of contexts.
2 Retrieve individual units’ activations. 














## Covariate shift oder so
There are three different types of dataset shifts:

<!-- https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/ -->
<!-- http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf -->
- Shift in the independent variables (Covariate Shift)
- Shift in the target variable (Prior probability shift)
- Shift in the relationship between the independent and the target variable (Concept Shift)

All three mentioned shifts could have a negative impact on the performance of a machine learning model, but this thesis focuses solely on the covariate shift. The term "covariate shift" defines changes in the distribution of the independent variables. [@covariante_shift] \*@fig:covariate_shift_regression illustrates the challenge if the training samples do not represent the test samples well in a regression problem due to either bad data acquisition or lousy choice of train/test splits.

![This illustration demonstrates the negative impact of the covariate shift on the success of the machine learning model trying to learn a true function (Red curve). Given the training samples (Blue dots), the model learns the linear learned function (Green line). The performance on the test samples (Black) will be terrible because the learned function does not approximate the true function very well in the space around the test samples. In an ideal setting, the training samples should have been equally distanced and scattered over the whole space of the true function with as low variance as possible. [[@covariate_shift_regression]](#references)](source/figures/covariate_shift_regression.png "Covariate shift demonstration for a regression problem"){#fig:covariate_shift_regression width=50%}

The challenge of facing a covariate shift in data distributions is a modality-independent problem. This phenomenon occurs in regression problems, natural language processing, computer vision, and other data representations. The universal language model BERT, which Google has developed, can understand sentences and generate suitable embeddings. A massive amount of data is used to create such a power model, which inevitably contains embedded biases. For example, a specific name always has a negative connotation, or certain words are associated with one gender over the other, independent of the context. [@bert_bias]

In a final example, which takes place in the context of a hospital, a patient could suffer dangerous consequences if such a covariate shift in a deployed machine-learning model remains undetected. A team of artificial intelligence (AI) researchers and radiologists claims to have successfully developed a machine-learning model which reliably detects COVID-19 from chest radiographs. However, experiments reveal that high accuracy is not achieved because of actual medical pathology features but because of confounding factors. In the worst possible scenario, a different hospital provides data with similar confounding factors due to the fact that they are using the same type of x-ray machine or other factors. These findings lead to an alarming situation where the machine learning model appears accurate but fails when tested in new hospitals. [@covid_shortcuts_over_signal]

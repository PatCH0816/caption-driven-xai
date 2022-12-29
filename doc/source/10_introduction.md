\setcounter{page}{1}
\pagenumbering{arabic}
\setlength{\parindent}{0.0in}

# Introduction
<!--- What is machine learning? How does it impact the world? -->
Machine learning is a collection of methods where programs are not explicitly programmed but learn from data instead. An increasing number of exciting machine learning applications are revolutionizing the modern world. Machine learning is used to predict traffic in Google Maps, recommend Movies on Netflix, assess the situation around self-driving cars, detect spam in E-Mails, etc. There is no doubt that many more exciting applications will come, but with great power comes great responsibility.

<!--- What is the problem? -->
With the ever-increasing power and responsibility of machine learning models at the core of many applications, they must prove their robustness. Robustness in AI addresses one of the most critical research areas in machine learning. "Robustness" refers to a model's ability to resist being fooled. Many machines learning beginner problems presented in school or one of the gazillion great online courses like "Specialization in Deep Learning" on coursera.org offer the huge benefit that the data in the classic train, validation, and test splits are typically sampled from the same data distribution. This is a massive advantage to getting an excellent introduction to the fascinating world of machine learning. Nevertheless, this differs in real-world applications in the sense that data distributions shift, and this leads to all kinds of challenges. There are three different types of dataset shifts:

<!-- https://www.analyticsvidhya.com/blog/2017/07/covariate-shift-the-hidden-problem-of-real-world-data-science/ -->
<!-- http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf -->
- Shift in the independent variables (Covariate Shift)
- Shift in the target variable (Prior probability shift)
- Shift in the relationship between the independent and the target variable (Concept Shift)

All three mentioned shifts could have a negative impact on the performance of a machine learning model, but this thesis focuses solely on the covariate shift. The term "covariate shift" defines changes in the distribution of the independent variables. [@covariante_shift] \*@fig:covariate_shift_regression illustrates the challenge if the training samples do not represent the test samples well in a regression problem due to either bad data acquisition or lousy choice of train/test splits.

![This illustration demonstrates the negative impact of the covariate shift on the success of the machine learning model trying to learn a true function (Red curve). Given the training samples (Blue dots), the model learns the linear learned function (Green line). The performance on the test samples (Black) will be terrible because the learned function does not approximate the true function very well in the space around the test samples. In an ideal setting, the training samples should have been equally distanced and scattered over the whole space of the true function with as low variance as possible. [[@covariate_shift_regression]](#references)](source/figures/covariate_shift_regression.png "Covariate shift demonstration for a regression problem"){#fig:covariate_shift_regression width=50%}

The challenge of facing a covariate shift in data distributions is a modality-independent problem. This phenomenon occurs in regression problems, natural language processing, computer vision, and other data representations. The universal language model BERT, which Google has developed, can understand sentences and generate suitable embeddings. A massive amount of data is used to create such a power model, which inevitably contains embedded biases. For example, a specific name always has a negative connotation, or certain words are associated with one gender over the other, independent of the context. [@bert_bias]

In a final example, which takes place in the context of a hospital, a patient could suffer dangerous consequences if such a covariate shift in a deployed machine-learning model remains undetected. A team of artificial intelligence (AI) researchers and radiologists claims to have successfully developed a machine-learning model which reliably detects COVID-19 from chest radiographs. However, experiments reveal that high accuracy is not achieved because of actual medical pathology features but because of confounding factors. In the worst possible scenario, a different hospital provides data with similar confounding factors due to the fact that they are using the same type of x-ray machine or other factors. These findings lead to an alarming situation where the machine learning model appears accurate but fails when tested in new hospitals. [@covid_shortcuts_over_signal]

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
As demonstrated, it is mission-critical to unearth hidden problems in real-world data science and not to fall for "correlation is not causation" problems. Accepting the fact that these challenges exist is the first step to improvement. At first, one needs to understand what a machine learning model is doing. The right tool for that kind of task are methods from the explainable AI toolbox. Getting back to the COVID-19 chest radiographs example, XAI supports AI systems to reduce the risk of learning undesired "shortcuts" instead of medically relevant pathology features. XAI is mainly used to understand and improve machine learning models, which suffer from low performance. Instead, XAI should be considered a prerequisite for all machine-learning models with significant responsibility, e.g. in healthcare environments. Hopefully, the vast amount of XAI methods will enable "Data Justice", "Data Trust" and "Data Fairness" in future applications.

<!--- Different state of the art approaches -->
<!--- grad-cam heatmaps [@xai_gianfagna_dicecco] -->
Part of this thesis is to work on computer vision problems. A widely used XAI method is generating saliency maps to understand which image region excites the machine-learning model the most for a specific class. Saliency maps highlight an area of pixels that contribute the most to the actual prediction. [@saliency_maps]

\*@fig:wolves_and_dogs_prediction demonstrates where saliency maps are helpful. The task is to classify the images into wolves and huskies. Five out of six predictions are correct. One question remains: Is this a good classifier?

![Shown is a binary classification task on six images of wolves and dogs. Five out of six predictions are correct. [[@wolves_and_dogs_prediction]](#references)](source/figures/wolf_or_husky.png "Wolf or husky predictions"){#fig:wolves_and_dogs_prediction width=100%}

\*@fig:husky_saliency_map demonstrates that the model in use did not use expected features like the colors of the fur, shape of the ears or the length of the snout to distinguish between wolves and huskies, but used the confounding factor "snow" from the background. The results from \*@fig:wolves_and_dogs_prediction suggests an accuracy of the model of about $\frac{5}{6} = 83\%$ to classify huskies and wolves, but in reality the model just learned to classify between "snow" and "no snow" and therefore failed to learn the task gracefully.

![The image (a) displays the image of a husky, which is classified as a wolf. The saliency map in image (b) provides a visual explanation that the model did not pay attention to the animal, but to the snow in the background. [[@wolves_and_dogs_xai]](#references)](source/figures/husky_saliency_map.png "Husky classified as wolf."){#fig:husky_saliency_map width=80%}

One of the most widespread XAI methods which produces saliency maps is the gradient class-activation-map (Grad-CAM) method. The Grad-CAM method uses the gradients of the classification score with respect to the final convolutional feature map to highlight the pixels in the input image with the most significant influence on the classification score. [@grad_cam] This Grad-CAM method is considered incredibly useful for comprehending what the machine learning model is doing. However, a common disadvantage in all saliency map producing methods is that understanding where the machine-learning model focuses does not tell what it is doing with that region of interest. Therefore, saliency maps do not reveal what the model is thinking but where it is looking only.

<!-- #TODO: Show table of trust in model before and after the explanation -->

<!-- Broden dataset
http://netdissect.csail.mit.edu/ (Paper)
https://medium.com/analytics-vidhya/demystifying-hidden-units-in-neural-networks-through-network-dissection-7d3ac657c428 (Simplified explanation of the paper in form of a blog) -->
How to discover concepts learned in the internal activations of models. The idea is to apply the model to a very diverse dataset called Broden and keep track of the highest activations of each neuron to each of the images. [@network_dissection]

<!--- What is our solution approach? -->
<!--- Describe the idea -->
The new XAI method presented in this thesis addresses the previously mentioned problems and attempts to solve them differently. Saliency maps highlight specific areas of interest in images but do not reveal what the machine-learning model thinks about this information. In an idealized world, the model would tell in written text what it sees in the image. The new XAI method presented in this thesis attempts to obtain a text-based explanation for a given machine-learning model. A biased dataset demonstrates that this new XAI method is working as intended. This biased dataset contains a covariate shift between the train/validation and test sets. The objective of the novel XAI text-based method is to reveal that the model under test focuses on bias instead of learning the actual task.

<!--- Overview chapters -->
The development of the novel XAI approach involves many different components. Each component is documented in detail in one of the following chapters:

- \*@sec:problem-description defines current challenges and opportunities in the world of robust machine learning. The original idea, which provides the starting point for this project and its milestones are also included.
- \*@sec:dataset introduces the used dataset to train, validate and test the model under test. Furthermore, the purposely introduced bias in the dataset is explained in detail. 
- \*@sec:contrastive-language-image-pre-training explains what the contrastive language image pre-training (CLIP) model is and how it works. CLIP is a core component of this new XAI method.
- \*@sec:network-surgery provides an overview of how this novel XAI method works. All involved components are explained in detail.
- \*@sec:results evaluates the performance which allows for a discussion on the suitability of the new XAI method for a given situation.
- \*@sec:conclusion consolidates all ideas from the previous chapters, summarizes the gained knowledge from this project, discusses open questions and shares some advice on future approaches on this topic.
- \*@sec:closing-words contains a personal reflection of this thesis from the author's point of view.

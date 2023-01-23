\setcounter{page}{1}
\pagenumbering{arabic}
\setlength{\parindent}{0.0in}

# Introduction
<!--- What is machine learning? How does it impact the world? -->
The machine learning toolbox is a collection of methods where models are not explicitly programmed but learn from data instead. An increasing number of exciting machine learning applications are disrupting the modern world. Machine learning is used to predict traffic in Google Maps, recommend Movies on Netflix, assess the situation around self-driving cars, detect spam in E-Mails, etc. There is no doubt that many more exciting applications will come, but with great power comes great responsibility.

<!--- What is the problem? -->
<!-- Robustness: https://vectorinstitute.ai/2022/03/29/machine-learning-robustness-new-challenges-and-approaches/ -->
<!-- "Robustness" refers to a model's ability to resist being fooled. -->
With the ever-increasing power and responsibility of machine learning models at the core of many applications, they must prove their robustness. Robustness in AI addresses one of the most critical research areas in machine learning. The performance of a robust model is allowed to deviate just a little bit when using real-world data instead of training data. This definition is why every machine-learning development process involves dividing the dataset into training, validation and test splits. The training split selects a suitable machine-learning model with a set of initial hyperparameters. The validation split tunes the hyperparameter to reach a performance optimum. The performance on the test split is the final result to be published. Many machines learning beginner problems presented in school or one of the gazillion great online courses like "Specialization in Deep Learning" on coursera.org and others offer one huge benefit. Typically, the data for the classic training, validation and test splits are sampled from the same data distribution. This property of the sampling process is a massive advantage to getting an excellent introduction to the fascinating world of machine learning. Nevertheless, in the real-world scenario, there is always a risk involved that the data used for the training, validation and test splits does not accurately reflect the data obtained by the deployed model. This distribution shift between the data used for the development of the model and the deployed model is called a covariate shift. A covariate shift hides the dangerous problem that a model seems to work in the lab for its intended task. It achieves excellent performance in the training, validation and test splits but fails in the real world.

<!--- What is our solution approach? -->
<!--- Describe the idea -->
One obvious solution to this problem is to ensure that the data for the development of the model 100% reflects real-world data. Since this is not an easy task, this work presents a new XAI that tackles this problem differently. In an idealized world, the model would tell in written text what it sees in the image. The new XAI method presented in this work attempts to obtain a caption-based explanation for a given machine-learning model. A biased dataset demonstrates that this new XAI method is working as intended. This biased dataset contains a covariate shift between the train/validation/test datasets (Simulating the available data during the model development) and the real-world set (Simulating real-world data after deployment). The objective of the novel XAI caption-based method is to reveal that the model under test focuses on bias instead of learning the actual task. This finding enables the developer to improve the model and increase the machine-vision model's robustness before deploying it in the real world.

<!--- Overview chapters -->
The development of the novel XAI approach involves many different components. The following chapters include a detailed description of all involved components:

- \*@sec:problem-description defines current challenges and opportunities in the world of robust machine learning. The original idea, which provides this project's starting point and milestones, is also included.
- \*@sec:explainable-artificial-intelligence is focused on existing explainable AI methods. Their advantages and disadvantages are discussed.
- \*@sec:contrastive-language-image-pre-training explains the contrastive language-image pre-training (CLIP) model and how it works. CLIP is a core component of this novel XAI method.
- \*@sec:dataset introduces the used dataset to train, validate and test the model under test. Furthermore, the purposely introduced bias in the dataset is explained in detail. 
- \*@sec:standalone-resnet-model presents the selection process for a suitable machine-learning model to be fooled. The training, validation and test splits performance demonstrate a fooled standalone ResNet model, which is a perfect candidate to demonstrate the novel XAI method.
- \*@sec:novel-xai-method provides an overview of how this novel XAI method works. All involved components are explained in detail.
- \*@sec:results evaluates the performance, which allows for a discussion on the suitability of the new XAI method for a given situation.
- \*@sec:conclusion consolidates all ideas from the previous chapters, summarizes the gained knowledge from this project, discusses open questions and shares some advice on future approaches on this topic.
- \*@sec:closing-words contains a personal reflection of this work from the author's point of view.

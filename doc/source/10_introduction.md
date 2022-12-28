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

In a final example, which takes place in the context of a hospital, a patient could suffer dangerous consequences if such a covariate shift in a deployed machine-learning model remains undetected. A team of artificial intelligence (AI) researchers and radiologists claims to have successfully developed a machine-learning model which reliably detects COVID-19 from chest radiographs. However, experiments reveal that high accuracy is not achieved because of actual medical pathology features but because of confounding factors. These findings lead to an alarming situation where the machine learning model appears accurate but fails when tested in new hospitals. [@covid_shortcuts_over_signal]

<!--- What is an explaination method? Why is it needed? -->
insight is the first step to improvement
XAI can help
Unearthing hidden problems in Real World Data Science
husky example
In addition, we show that evaluation of a model on external data is insufficient to ensure AI systems rely on medically relevant pathology, because the undesired ‘shortcuts’ learned by AI systems may not impair performance in new hospitals.
These findings demonstrate that explainable AI should be seen as a prerequisite to clinical deployment of machine-learning healthcare models.

<!--- Different approaches -->
<!--- State of the art -->
gram-cam heatmaps

How to discover concepts learned in the internal activations of models. The idea is to apply the model to a very diverse dataset called Broden and keep track of the highest activations of each neuron to each of the images.
http://netdissect.csail.mit.edu/ (Paper)
https://medium.com/analytics-vidhya/demystifying-hidden-units-in-neural-networks-through-network-dissection-7d3ac657c428 (Simplified explanation of the paper in form of a blog)

<!--- What is our solution approach? -->
use clip to get an explanation

<!--- Describe the idea -->
how does the idea work?

<!--- Overview chapters -->
This thesis describes the development of a novel, explainable AI approach, which expresses the characteristics of a machine learning model in a text-based format. Each step to obtain the final result is documented in detail in one of the following chapters:

- \*@sec:problem-description defines current challenges and opportunities in the world of robust machine learning. The original idea, which provides the starting point for this project and its milestones are also included.
- sdfg
- \*@sec:closing-words contains a personal reflection of this thesis from the author's point of view.

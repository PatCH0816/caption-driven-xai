# Dataset
Machine learning models learn information from datasets without relying on predetermined equations as a model. This chapter introduces the original dataset and its customization used for this work to demonstrate the caption based explainable AI method.

## Original dataset
The custom dataset uses a modified version of the Modified National Institute of Standards and Technology (MNIST) dataset of handwritten digits. [@mnist_dataset] The original MNIST dataset consists of 60'000 training samples and 10'000 test samples. Each sample is a 28x28 pixel grayscale image of digits from 0 to 9. This dataset is commonly used as a benchmark to assess the performance of a new multiclass classification model.

## CLIP limitations
Since the contrastive language-image pre-training (CLIP) architecture is a fundamental part of the caption based explainable AI method, the proof of concept must fit its capabilities. Applying CLIP to the classic MNIST problem to classify handwritten digits from 0 to 9 reveals one of its weaknesses. As discussed in \*@sec:contrastive-language-image-pre-training, CLIP performs well for very different concepts, like distinguishing between cars and horses. CLIP struggles with differentiating between car models since all car models share the same "car" concept. Following the keep-it-simple-and-stupid (KISS) approach, CLIP is applied to the MNIST dataset to assess its performance in this task using a ResNet-50 image encoder. The confusion matrix in \*@fig:clip_mnist_multiclass summarizes the modest, normalized results.

![Relative class accuracies of CLIP applied to MNIST handwritten digits multiclass classification task.](source/figures/clip_mnist_multiclass.png "Class accuracies of CLIP applied to MNIST."){#fig:clip_mnist_multiclass width=90%}

## Custom dataset
<!-- Multiclass problem to binary problem -->
CLIP cannot classify all digits in the MNIST dataset with satisfying accuracy. Following the KISS approach, the multiclass problem is reduced to a binary classification problem of the two "best" digits to classify, which are the digits 5 (Accuracy: 97%) and 8 (Accuracy: 87%), as demonstrated in \*@fig:clip_mnist_multiclass. This observation is surprising since these digits look somehow similar. No further analysis is carried out regarding this observation because CLIP's custom dataset is proprietary.

<!-- preprocessor -->
Every image encoder of a CLIP configuration comes with its preprocessor. Detailed information about the preprocessor can be found in \*@tbl:clip_rn50_preprocess in the appendix. This preprocessor extends the 28x28 grayscale MNIST images to 3x28x28 images.

<!-- Add bias -->
This filtering and preprocessing leads to a modified MNIST subset with the digits 5 and 8 exclusively. The pre-processed original dataset with 70'000 samples is balanced (Upsampling, if there are less samples available than desired, downsampling otherwise) and split into training (5'000 samples for each digit), test dataset (1'000 samples for each digit) and real-world dataset (1'000 samples for each digit). During development, the state-of-the-art (SOTA) k-fold-cross-validation approach divides the training dataset into training and validation folds for k-times to train k-models as shown in \*@fig:dataset_split for k=5.

![The original MNIST dataset is divided into training, test and real-world sub-datasets. Using 5-fold cross-validation, the training dataset is further divided into a training and validation dataset to detect potential selection bias or overfitting problems.](source/figures/dataset_split.png "Dataset splits into training, validation, test and real-world datasets."){#fig:dataset_split width=100%}

This SOTA method reveals potential selection bias or overfitting problems and robustly estimates the model's performance. These training, validation and test splits are available during the development of a model. After deploying the model, it gets to solve its designated task in the real-world environment. The real-world dataset simulates this deployment behavior. Therefore, we have four datasets at hand:

- Training dataset (4 out of 5 folds for crossvalidation from original training dataset)
- Validation dataset (1 out of 5 folds for crossvalidation from original training dataset)
- Test dataset (From same data distribution as the training and validation datasets)
- Real-world dataset (From different data distribution as the training, validation and test datasets)

The common training, validation and test datasets are used to create a familiar training, validation and test learning curves plot. The objective of the real-world dataset is to demonstrate the challenge which arises, if the data distribution during the development is not a good match to a different data distribution in the real-world. Like the saying, "The difference between theory and practice is larger in practice than in theory", the data distributions between the available data during development (training, validation and test datasets) and in the real environment (real-world dataset) are most likely different. In the worst possible situation, a highly correlated bias is involved during the development process, which causes a covariate shift. There is a lovely definition by Steffen Bickel et al., which goes like this: "We address classification problems for which the training instances are governed by an input distribution that is allowed to differ arbitrarily from the distribution—problems also referred to as classification under covariate shift". [@covariate_shift]

<!-- Color channels -->
The introduced covariate shift adds a different color channel assignment between the datasets available during the development of the model (training, validation and test datasets) and the simulated deployment (real-world dataset). The original grayscale image is mapped to the red or green color channel according to the digit associated with the image. This color feature is designed to create a perfect correlation between the color channels and the labels in the development datasets (Training, validation and test datasets). A model trained on the training dataset has two sources of information to learn from:

- Classify the digits by their color. (Undesired in general, because the model falls for the bias and ignores the shape of the digit)
- Classify the digits by their shape. (Desired in general because the model ignores the bias and pays attention to the shape of the digit)

The hypothesis is that it is easier for a model to classify digits according to their colors instead of shapes. Therefore, a model would probably fall for the undesired color bias instead of the desired shape feature. If a model learns to classify the digits according to their color features, the developer observes satisfying training, validation and test results. Therefore, he will be confident that the models recognize digits as intended. As soon as the model is deployed into the real-world (Simulate with the real-world dataset), the model fails miserably because the colors no longer correlate with the numbers. This different behavior of the color feature between the training/validation datasets and the test dataset is called covariate shift. At this point, it is essential to mention that this digit recognition task is a proxy for any important real-world task where a model could be deployed and fail miserably, as described in the coronavirus disease of 2019 (COVID-19) chest x-ray example in \*@sec:covariate-shift.

In the training, validation and test datasets (Simulating the available data during the model development), all digits with the value 5 are red and all digits with the value 8 are green.

![All digits with the value 5 are red and all digits with the value 8 are green in the training, validation and test dataset.](source/figures/dataset_train_val_test.png "An example of each of the two digits 5/8 contained in the training, validation and test dataset."){#fig:modified_mnist_development width=90%}

In the real-world dataset (Simulating real-world data after deployment of the model), the colors of the digits are assigned at random.

![The colors of the digits are assigned at random in the real-world dataset.](source/figures/dataset_real_world.png "An example of each of the two digits 5/8 contained in the real-world dataset."){#fig:modified_mnist_real_world width=90%}

## Theory summary
The core of the idea is that a model trained on this custom dataset will focus on the undesired correlating feature (Color of the digits) instead of the desired causating feature (Shape of the feature) to classify the digits. This bias leads to high accuracy on the training, validation and test dataset during the development of a new model and terrible accuracy on the real-world dataset. The presented caption based explainable AI method should then be able to reveal the problem!

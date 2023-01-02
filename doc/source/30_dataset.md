# Dataset
One thing all machine learning models have in common is the need for data they are learning from. This chapter introduces the dataset used for this thesis to develop the novel XAI method.

## Original dataset
The custom dataset uses a modified version of the Modified National Institute of Standards and Technology (MNIST) dataset of handwritten digits. [@mnist_dataset] The original MNIST dataset consists of 60'000 training samples and 10'000 test samples. Each sample is a 28x28 pixel grayscale image of digits from 0 to 9. This dataset is commonly used as a benchmark to assess the performance of a new multiclass classification model.

## CLIP limitations
Since the contrastive language image pre-training (CLIP) architecture is a fundamental part of the novel XAI method, the proof of concept must fit its capabilities. Applying CLIP to the classic MNIST problem to classify handwritten digits from 0 to 9 reveals one of its weaknesses. As discussed in \*@sec:contrastive-language-image-pre-training, CLIP performs well for very different concepts, like distinguishing between cars and horses. CLIP struggles with differentiating between car models since all car models share the same "car" concept. Following the keep-it-simple-and-stupid (KISS) approach, CLIP is applied to the MNIST dataset to check its performance in this task using a ResNet-50 image encoder. The confusion matrix in \*@fig:clip_mnist_multiclass summarizes the disappointing results.

![Class accuracies of CLIP applied to MNIST handwritten digits multiclass classification task.](source/figures/clip_mnist_multiclass.png "Class accuracies of CLIP applied to MNIST."){#fig:clip_mnist_multiclass width=90%}

## Modified dataset
CLIP cannot classify all digits in the MNIST dataset with satisfying accuracy. Following the keep-it-simple-and-stupid approach, the multiclass problem is reduced to a binary classification problem of the two "best" digits to classify, which are the digits 5 (Accuracy: 97%) and 8 (Accuracy: 87%). This observation is surprising since these digits look somehow similar. No further analysis is carried out regarding this observation because CLIP's custom dataset is proprietary.

Nevertheless, this filtering leads to an MNIST subset with the digits 5 and 8 exclusively. Usually, this dataset would be balanced and split into a training, validation and test dataset. A more challenging dataset is needed to demonstrate the novel XAI method's performance. A challenging dataset in this context means adding a bias, such that the test dataset does not have the same characteristics as the training and validation datasets. This shift in their distributions is called a covariate shift. There is a lovely definition by Steffen Bickel et al., which goes like this: "We address classification problems for which the training instances are governed by an input distribution that is allowed to differ arbitrarily from the test distribution—problems also referred to as classification under covariate shift". [@covariate_shift] The introduced covariate shift adds a different color channel assignment between the datasets. Introducing the color channels extends the shapes of the images from 28x28 pixels to  28x28x3 pixels. The original grayscale image is mapped to the red or green color channel according to the digit associated with the image. This color feature results in a high correlation between the color channels and the labels. A model trained on the training dataset has two choices:

- Classify the digits by their color. (Undesired, because the model falls for the bias and ignores the shape of the digit)
- Classify the digits by their shape. (Desired, because the model ignores the bias and pays attention to the shape of the digit)

The hypothesis is that it is easier for a model to classify digits according to their colors instead of shapes. Therefore, a model would fall for the undesired color bias instead of the desired shape feature. If a model learns to classify the digits according to their color features, it will miserably fail the test dataset with the mixed-up colors. This different behavior of the color feature between the training/validation datasets and the test dataset is called covariate shift.

The modified training and validation datasets have the following properties:
- All digits 5 are colored in red.
- All digits 8 are colored in green.

The modified test dataset has the following properties:
- All digits 5 are colored in green.
- All digits 8 are colored in red.

#TODO do not write bias! you are describing a covariate shift! search and replace in whole thesis carefully! covariate shift is the problem, bias the effect

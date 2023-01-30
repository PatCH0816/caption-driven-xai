# Results
The most important question of the presented work is: "Is it reasonable to use CLIP to explain a ResNet-50 neural network?". This work presents a caption-based explainable AI model introduced in \*@sec:caption-based-explainable-ai to answer this question. The model to be explained is built using a ResNet-50 architecture as specified in \*@sec:standalone-model and trained on the dataset introduced in \*@sec:dataset. The following sections dive deep into the intermediate steps and demonstrate the results of the caption-based explainable AI model.

## Initial situation
The ResNet-50 standalone model introduced in \*@sec:standalone-model is trained on the dataset introduced in \*@sec:dataset. The training, validation and test learning curves indicate a low bias, low variance model as shown in \*@fig:is_performance_biased_without_test_fool. A machine learning developer could be tempted to deploy this model into the real-world environment.

![This figure illustrates the low bias, low variance learning progress of the transfer learned ResNet-50 model on the biased color MNIST training, validation and test datasets during the model development process.](source/figures/performance_biased_without_test_fool.png "Training, validation and test learning curves from standalone ResNet-50 on custom MNIST dataset for binary classification."){#fig:is_performance_biased_without_test_fool width=50%}

If the ResNet-50 standalone model introduced in \*@sec:standalone-model shown would be deployed into the real-world environment, its accuracy would drop significantly and demonstrate random behavior as shown in \*@fig:is_performance_biased_with_test_fool. The caption-based explainable AI model should support machine learning developers to detect such a problem before deployment. The findings from the caption-based explainable AI model can then be used to fix the standalone model and therefore improve its robustness.

![This figure illustrates the low bias, low variance learning progress of the transfer learned ResNet-50 model on the biased color MNIST training, validation and test datasets during the model development process. Additionally, the real-world curve demonstrates the poor performance simulated in the real-world environment.](source/figures/performance_biased_with_test_fool.png "Training, validation, test and real-world learning curves from standalone ResNet-50 on custom MNIST dataset for binary classification."){#fig:is_performance_biased_with_test_fool width=50%}

## caption-based explainable AI
Work in progress..

![Gaussian activations statistics from random layer 39 of the standalone model. Activations from other layers are also Gaussian.](source/figures/activations_gaussian_mean_std_standalone.png "Gaussian activations statistics from random layer 39 of the standalone model."){#fig:activations_gaussian_mean_std_standalone width=75%}

![Gaussian activations statistics from random layer 39 of the clip model. Activations from other layers are also Gaussian.](source/figures/activations_gaussian_mean_std_clip.png "Gaussian activations statistics from random layer 39 of the clip model."){#fig:activations_gaussian_mean_std_clip width=75%}

## Remove bias and training
Work in progress..
<!-- 
![The training, validation and test curves indicate low bias and low variance on the grayscale MNIST dataset (Removed correlating color feature) during the retraining of the standalone model.](source/figures/performance_unbiased_without_test_fool.png "Training, validation and test curves on the grayscale MNIST dataset."){#fig:performance_unbiased_without_test_fool width=50%} -->

![The training, validation and test curves indicate low bias and low variance on the grayscale MNIST dataset (Removed correlating color feature) during the retraining of the standalone model. The real-world performance demonstrates that removing the color feature and retraining the standalone model helped to increase the model's robustness.](source/figures/performance_unbiased_with_test_fool.png "Training, validation, test and real-world curves on the grayscale MNIST dataset."){#fig:performance_unbiased_with_test_fool width=50%}

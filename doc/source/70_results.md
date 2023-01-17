# Results
text text text.

![The biased standalone ResNet-50 model consists of an actual image encoder and the fully-connected linear classifier. The training, validation and test curves indicate a low bias, low variance model regarding the dataset at hand during development. Without using XAI, the model could still be biased.](source/figures/abstract/abstract_1_situation.png "Architecture and training, validation and test curves of the biased standalone model."){#fig:performance_biased_without_test_fool width=100%}

text text text.

![The architecture of the novel XAI method uses the core architecture of CLIP. Using CLIP's text encoder (Purple) and image encoder (Green), the resulting embedding similarities reveal what the CLIP image encoder (Green) is focusing on using captions. The network surgery process allows integration of any standalone model into CLIP, so CLIP can explain what the image encoder (Red) from a standalone model focuses on.](source/figures/abstract/abstract_2_clip.png "Overview of the novel XAI method."){#fig:performance_biased_without_test_fool width=100%}

text text text.

![The novel XAI method reveals the color feature as a highly correlating bias in the dataset available during development. Removing the color feature using a pre-processor and retraining the model makes the standalone model more robust. The captions from CLIP's section of the new XAI method reveal that the feature relevant to the decision-making process shifts from the color to the shape feature.](source/figures/abstract/abstract_3_xai.png "Comparison between the standalone model with and without the use of XAI."){#fig:performance_biased_without_test_fool width=100%}

text text text.

![The training, validation and test curves indicate low bias and low variance on the grayscale MNIST dataset during the retraining of the standalone model.](source/figures/performance_unbiased_without_test_fool.png "Training, validation and test curves on the grayscale MNIST dataset."){#fig:performance_biased_without_test_fool width=50%}

text text text.

![The training, validation and test curves indicate low bias and low variance on the grayscale MNIST dataset during the retraining of the standalone model. The real-world performance demonstrates that removing the color feature and retraining the standalone model helped to increase the model's robustness.](source/figures/performance_unbiased_with_test_fool.png "Training, validation, test and real-world curves on the grayscale MNIST dataset."){#fig:performance_biased_without_test_fool width=50%}

# Results
Work in progress..

## Initial situation
Work in progress..

![The training, validation and test curves indicate low bias and low variance on the grayscale MNIST dataset (Removed correlating color feature) during the retraining of the standalone model.](source/figures/performance_unbiased_without_test_fool.png "Training, validation and test curves on the grayscale MNIST dataset."){#fig:performance_unbiased_without_test_fool width=50%}

## caption based explainable AI
Work in progress..

![Gaussian activations statistics from random layer 39 of the standalone model. Activations from other layers are also Gaussian.](source/figures/activations_gaussian_mean_std_standalone.png "Gaussian activations statistics from random layer 39 of the standalone model."){#fig:activations_gaussian_mean_std_standalone width=75%}

![Gaussian activations statistics from random layer 39 of the clip model. Activations from other layers are also Gaussian.](source/figures/activations_gaussian_mean_std_clip.png "Gaussian activations statistics from random layer 39 of the clip model."){#fig:activations_gaussian_mean_std_clip width=75%}

## Remove bias and training
Work in progress..

![The training, validation and test curves indicate low bias and low variance on the grayscale MNIST dataset (Removed correlating color feature) during the retraining of the standalone model. The real-world performance demonstrates that removing the color feature and retraining the standalone model helped to increase the model's robustness.](source/figures/performance_unbiased_with_test_fool.png "Training, validation, test and real-world curves on the grayscale MNIST dataset."){#fig:performance_unbiased_with_test_fool width=50%}

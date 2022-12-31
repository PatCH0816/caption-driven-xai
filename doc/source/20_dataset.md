# Dataset
- a dataset is needed to train a model
- suitable for supervised model training
## Original dataset
- mnist dataset is used
- what is the mnist dataset
- multiclass classification problem
[@mnist_dataset]

## CLIP limitations
- CLIP cannot distinguish between different numbers
- KISS: pick two best classes
- create binary classification problem

## Modified dataset
- add bias
- bias leads to covariate shift between train, validation and test datasets

We address classification problems for which the training instances are governed by an input distribution that is allowed to differ arbitrarily from the test distribution—problems also referred to
as classification under covariate shift.
https://jmlr.csail.mit.edu/papers/volume10/bickel09a/bickel09a.pdf

do not write bias! you are describing a covariate shift! search and replace in whole thesis carefully!

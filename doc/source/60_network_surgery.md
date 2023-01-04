# Network surgery
<!-- The two resnets are not the same! -->
<!-- #TODO Modified ResNet -->
<!-- ResNet architecture: https://towardsdatascience.com/understanding-and-visualizing-resnets-442284831be8 -->
<!-- ResNet expects input images of size: 224x224 -->
We selected a ResNet-50 as the model architecture. We
modified the base ResNet-50 with the anti-alias improvements from (Zhang, 2019) and used weight norm (Salimans & Kingma, 2016) instead of batch norm (Ioffe &
Szegedy, 2015) to avoid leaking information about duplicates via batch statistics - a problem previously noted in
(Henaff, 2020). We also found the GELU activation function (Hendrycks & Gimpel, 2016) to perform better for this
task. We trained the model with a total batch size of 1,712
for approximately 30 million images sampled from our pretraining dataset. At the end of training it achieves nearly
100% accuracy on its proxy training task.

layer switching:
- preserve CLIP embedding space
- transfer decision critical layers from given model to CLIP

#TODO create image of network surgery

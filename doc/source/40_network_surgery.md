# Network surgery
<!-- The two resnets are not the same! -->
<!-- #TODO Modified ResNet -->
We selected a ResNet-50 as the model architecture. We
modified the base ResNet-50 with the anti-alias improvements from (Zhang, 2019) and used weight norm (Salimans & Kingma, 2016) instead of batch norm (Ioffe &
Szegedy, 2015) to avoid leaking information about duplicates via batch statistics - a problem previously noted in
(Henaff, 2020). We also found the GELU activation function (Hendrycks & Gimpel, 2016) to perform better for this
task. We trained the model with a total batch size of 1,712
for approximately 30 million images sampled from our pretraining dataset. At the end of training it achieves nearly
100% accuracy on its proxy training task.


- Researcher at MIT Computer Science and Artificial Intelligence Laboratory (CSAIL)
- What is going on inside of a neural network?
- Discover concepts learned by internal activations of models
- Broden dataset is a diverse dataset. 
- Idea to keep track of all activations of each neuron to each of the images
- In other words, this method interprets networks by providing meaningful labels to their hidden units.
- In the past, observations of hidden units have shown that human-interpretable concepts sometimes emerge in individual units within networks.
- Human-interpretable concepts include low-level concepts like colors and high-level concepts such as objects. By measuring the concept that best matches each unit, Net Dissection can break down the types of concepts represented in a layer.
1 The Broadly and Densely Labeled Dataset (Broden) unifies several densely labeled image data sets: ADE , Open Surfaces , Pascal-Context , Pascal-Part and Describable Textures Dataset. These data sets contain examples of a broad range of objects, scenes, object parts, textures, and materials in a variety of contexts.
2 Retrieve individual units’ activations. 
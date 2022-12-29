# Network surgery
<!-- #TODO: Broden dataset
http://netdissect.csail.mit.edu/ (Paper)
https://medium.com/analytics-vidhya/demystifying-hidden-units-in-neural-networks-through-network-dissection-7d3ac657c428 (Simplified explanation of the paper in form of a blog) -->
How to discover concepts learned in the internal activations of models. The idea is to apply the model to a very diverse dataset called Broden and keep track of the highest activations of each neuron to each of the images. [@network_dissection]

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
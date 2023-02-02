# Conclusion
<!-- 
- not merely a summary of the main topics covered or a re-statement of your research problem, but a synthesis of key points
- Recap what you did. In about one paragraph recap what your research question was and how you tackled it. Highlight the big accomplishments. Spend another paragraph explaining the highlights of your results. These are the main results you want the reader to remember after they put down the paper, so ignore any small details.
- Finally, finish off with a sentence or two that wraps up your paper. I find this can often be the hardest part to write. You want the paper to feel finished after they read these. One way to do this, is to try and tie your research to the “real world.” 
-->
This work successfully introduces a novel approach called the "Caption-based explainable AI method" to explain convolutional neural networks. Using a novel network surgery method, a standalone model to be explained is incorporated into CLIP. The resulting XAI model can identify the dominant concept that contributes the most to the model's predictions. The most promising result is the superiority of the novel XAI method over saliency maps in specific situations. The central thesis validated by this work is that a deeper understanding of the dominant concepts in convolutional neural networks is fundamental and can ultimately improve the model's robustness. Our findings suggest that this novel XAI method should not just be seen as a pure debugging tool but as a necessary prerequisite before deploying any machine vision convolutional neural network model.

The universal no-free-lunch theorem also applies to the caption-based explainable AI method. Therefore, we want to discuss the advantages and the disadvantages of the caption-based explainable AI method and the lovely story of the black swan, which is a real-life story of the covariate shift in the following sections:

\noindent
**Advantages**
	
- The main advantage is that the caption-based explainable AI method can identify the dominant concept of a standalone model to be explained. Suppose the identified dominant concept does align with the overall objective of the machine learning task. The finding indicates that the standalone model learned its desired task and can be considered robust.
- If the identified dominant concept differs from the overall objective, a bias has been identified. This finding opens the opportunity to remove the bias from the dataset and train an unbiased version of the standalone model to improve its robustness.
- Comparing the caption-based explainable AI method to saliency maps demonstrates the caption-based explainable AI method's supremacy in settings where the decision-critical features overlap correlating features. Applying saliency maps to the colored MNIST problem highlights the digit without any information if the digit's color or shape is the dominant concept. On the other hand, the caption-based explainable AI method can distinguish between the color and shape concepts of the digits if suitable captions are provided.

\noindent
**Disadvantages**

- Engineering suitable captions is a science in itself. Regarding captions, since CLIP is the core component of the caption-based explainable AI method, only concepts known to CLIP should be used in the captions. Captions describing concepts not present in CLIP's space of concepts will not result in the expected performance. This disadvantage is particularly concerning since the custom dataset CLIP has been trained on remains proprietary. Thankfully, the open-CLIP open-source implementation trained on the LAION dataset addresses this problem.
- While the caption-based explainable AI method can detect the highly correlated color bias used in the modified MNIST dataset, the cosine similarities did not change as much as expected. Part of the explanation is that only $\frac{3840}{22720} = 16.9\%$ of all activation maps from the standalone model were integrated into the caption-based explainable AI model using network surgery to maintain CLIP's concept space. This raises the question if it is possible to identify biases with a low correlation to the ground truth labels.
- CLIP would need to be finetuned for specific problems like recognizing diseases on x-rays.
- The caption-based explainable AI method is continuously focusing on the whole image. For busy images with several concepts, like a note with the text "iPad" stuck to an apple, the results could be ambiguous depending on the dominant feature.

\noindent
**Black swan**  
The black swan is a metaphor for the unknown and unexpected. In the old world, most European people believed that all swans were white. Every swan they have ever seen has been white. Therefore, they were convinced that their theory was true. Not until the Europeans found Australia they realized that there were black swans too. Trying hard to disprove a theory is essential to the scientific method. It is when the theories cannot be disproven that there is something genuine about our reality. This analogy also applies to machine learning scientists who face low bias and low variance models, which could still suffer from a covariate shift despite their excellent training, validation and test learning curves.

## Future work
As for most real-world projects, there is almost no finishing line. This section includes an ordered list of tasks in terms of complexity, which provides a good starting point for interested researchers to keep on researching on the caption-based explainable AI method.

- Find cosine similarities near 0 and 1 (Usually, the observed range during this work was $cos(\theta) = [0.05, 0.35]$.
- This work demonstrated the novel XAI method's feasibility detecting highly correlated biases. How about detecting biases with low correlation? Is it possible to prove that the model is bias-free?
- Apply the novel XAI method to more challenging datasets like the Stanford street view house numbers or the synthetic digits dataset. [@stanford_housenumbers] [@synthetic_digits]
- Replace the trivial binary classification task of the standalone model with a more challenging multiclass classification task.
- Add a moving window over the image to identify concepts in different locations with linear boundaries.
- Segment the image before identifying concepts in different locations with non-linear boundaries.
- Find a better loss function than the symetric cross entropy loss function to use the whole space of concepts, not just a fraction of the unit sphere. Due to the resulting larger space between concepts, the danger of running into numerical problems is reduced.
- Define the concepts a standalone model should focus on using captions. Treat the cosine similarities as the loss function value for a minimization problem, which trains the standalone model using the set of captions.
- Try to improve the activation matching process by introducing a threshold hyperparameter to obtain binarized activation maps, which can be compared and matched using an intersection over union score as in network dissection.

## Outlook
<!-- The greatest deception men suffer is from their own opinions. Leonardo da Vinci -->
<!-- Make the last 1/2 sentences memorable. -->
The proof of concept of the caption-based explainable AI method presented in this paper looks promising for future applications. The long-term vision for this novel XAI method is that it could help e.g. to remove social biases from existing machine vision applications. This would improve the trust in machine learning models and ultimately enhance our lives. As a general takeaway, the next time you see a low bias, low variance model, do not just assume it is an excellent model but ask yourself if you are suffering from the confirmation bias as explained in the story of the black swan. The caption-based explainable AI method may come to the rescue and identify the dominant concept, which will support you in making your model more robust.

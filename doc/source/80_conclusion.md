# Conclusion
<!-- 
- not merely a summary of the main topics covered or a re-statement of your research problem, but a synthesis of key points
- Recap what you did. In about one paragraph recap what your research question was and how you tackled it. Highlight the big accomplishments. Spend another paragraph explaining the highlights of your results. These are the main results you want the reader to remember after they put down the paper, so ignore any small details.
- Finally, finish off with a sentence or two that wraps up your paper. I find this can often be the hardest part to write. You want the paper to feel finished after they read these. One way to do this, is to try and tie your research to the “real world.” 
-->
<!-- The conclusion of a conclusion should: Restate your topic and why it is important. Restate your thesis/claim. Address opposing viewpoints and explain why readers should align with your position. -->
<!-- Recap what you did. In about one paragraph recap what your research question was and how you tackled it.
Highlight the big accomplishments. Spend another paragraph explaining the highlights of your results. These are the main results you want the reader to remember after they put down the paper, so ignore any small details.
Conclude. Finally, finish off with a sentence or two that wraps up your paper. I find this can often be the hardest part to write. You want the paper to feel finished after they read these. One way to do this, is to try and tie your research to the “real world.” Can you somehow relate how your research is important outside of academia? Or, if your results leave you with a big question, finish with that. Put it out there for the reader to think about to.
Optional Before you conclude, if you don’t have a future work section, put in a paragraph detailing the questions you think arise from the work and where you think researchers need to be looking next. -->
This work introduces a new approach called the caption-based explainable artificial intelligence (XAI) method to explain convolutional neural networks. Using a novel network surgery method, a standalone model to be explained is incorporated into CLIP. The resulting XAI model can identify the dominant concept that contributes the most to the model's predictions. The most promising result is the superiority of the novel XAI method over saliency maps in situations where spurious and salient features are present in overlapping pixel spaces. The central thesis validated by this work is that a deeper understanding of the dominant concepts in convolutional neural networks is fundamental and can be used to improve the model's robustness. Our findings suggest that this novel XAI method should not just be seen as a pure debugging tool but as a necessary prerequisite before deploying any machine vision convolutional neural network model.

\noindent
**Advantages**

- The main advantage is that the caption-based explainable AI method can identify the dominant concept of a standalone model to be explained. Suppose the identified dominant concept does align with the overall objective of the machine learning task. This alignment would confirm that the standalone model learned its desired task and can be considered robust.
- If the identified dominant concept differs from the overall objective, a bias has been identified. This finding opens the opportunity to remove the bias from the dataset and train an unbiased and robust version of the standalone model.
- Comparing the caption-based explainable AI method to saliency maps demonstrates the caption-based explainable AI method's supremacy in settings where the salient features overlap spurious features. Analyzing the Grad-CAM saliency maps of the colored MNIST problem highlights salient features of the digits without any information if the digit's color or shape is the dominant concept. On the other hand, the caption-based explainable AI method can distinguish between the color and shape concepts of the digits if suitable captions are provided.

\noindent
**Disadvantages**

- Since CLIP is the core component of the caption-based explainable AI method, only concepts known to CLIP should be used for the captions. Captions describing concepts not present in CLIP's space of concepts will not result in the expected performance. This disadvantage is particularly concerning because of CLIP's proprietary custom dataset. Thankfully, the open-CLIP open-source implementation trained on the public LAION dataset addresses this problem.
- While the caption-based explainable AI method can detect the spurious color feature present in the modified MNIST dataset, the cosine similarities do not change as significantly as expected. Part of the explanation is that only $16.9\%$ of all activation maps from the standalone model were integrated into the caption-based explainable AI model using network surgery to maintain CLIP's concept space. This raises the question if it is possible to identify biases with little correlation compared to the ground truth labels.
- CLIP needs to be finetuned for specific problems like recognizing diseases on x-rays.
- The caption-based explainable AI method focuses on the whole image. For busy images with several concepts, like a note with the text "iPad" stuck to an apple, the results could be misleading depending on the dominant feature being the apple or the note.

## Future work
<!-- The future work section is a place for you to explain to your readers where you think the results can lead you. What do you think are the next steps to take? What other questions do your results raise? Do you think certain paths seem to be more promising than others? -->
<!-- https://github.com/mlfoundations/open_clip/discussions/361 -->
As for most real-world projects, there is rarely a finishing line. While working on this project, a lot of promising leads arose that show great potential for further research and investigation: 

- Find cosine similarities near 0 and 1 (Usually, the observed range during this work was $cos(\theta) = [0.05, 0.35]$.
- This work demonstrates the novel XAI method's feasibility in highly correlated features. If the model can detect low correlation biases, its value as a scientific tool will increase drastically. 
- Apply the novel XAI method to more challenging datasets like the Stanford street view house numbers, synthetic digits and non-digit related datasets. Pushing the limits of the method could further outline potential business cases. 
- Replace the trivial binary classification task with a more challenging multiclass classification task.
- Add a moving window over the image to identify concepts in different locations of the image with linear boundaries.
- Segment the image before identifying concepts in different locations of the image with non-linear boundaries.
- Find a better loss function than the symmetric cross-entropy loss function to use the whole space of concepts, not just a fraction of the unit sphere. Due to the resulting larger space between concepts, the danger of running into numerical problems is reduced.
- Use captions for a language-guided training process. Treat the cosine similarities as the loss function value to align the image embedding with the text embedding during the training.
- Try to improve the activation matching process by introducing a threshold hyperparameter to obtain binarized activation maps, which can be compared and matched using an intersection over union score as in the network dissection XAI method.

\newpage
## Outlook
<!-- The greatest deception men suffer is from their own opinions. Leonardo da Vinci -->
<!-- Make the last 1/2 sentences memorable. -->
This work's proof of concept of the caption-based XAI method motivates further research. With some additional effort in this research area, we could reduce the risk of an ML model working in the lab but failing in the real-world. If further research succeeds, we could consider positioning the caption-based explainable artificial intelligence method as an essential tool in the future to promote safety, increase trust in ML models and enhance our lives.

We want to finish with a short story about the black swan to recall the challenge of the covariate shift in everyday situations. The black swan is a metaphor for the unknown and unexpected. In the old world, most European people believed that all swans were white. Every mature swan they have ever seen has been white. Therefore, they were convinced that their theory was true. Not until the Europeans discovered Australia they also realized that there were black swans. Trying to disprove a theory is essential to the scientific method. Only when all attempts to disprove a theory fail is the theory considered to depict reality accurately. This metaphor also applies to machine learning scientists who face low bias and low variance models, which could still suffer from a covariate shift despite their excellent training, validation and test learning curves. Remembering this story may help to recognize a covariate shift between the data available during the development of an ML model and deployment into the real-world environment. As a general takeaway, the next time you see a low bias, low variance model, do not just assume it is an excellent model but ask yourself if you are suffering from the confirmation bias as explained in the story of the black swan. The caption-based explainable AI method may come to the rescue and identify the dominant concept, which will support you in making your model more robust.

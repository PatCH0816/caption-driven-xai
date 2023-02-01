# Conclusion
This work successfully introduces a novel approach called the "Caption-based explainable AI method" to explain convolutional neural networks. Using a novel network surgery method, a standalone model to be explained is incorporated into CLIP. The resulting XAI model can identify the dominant concept that contributes the most to the model's predictions. The most promising result is the superiority of the novel XAI method over saliency maps in specific situations. The central thesis validated by this work is that a deeper understanding of the dominant concepts in convolutional neural networks is fundamental and can ultimately improve the model's robustness. Our findings suggest that this novel XAI method should not just be seen as a pure debugging tool but as a necessary prerequisite before deploying any machine vision convolutional neural network model.

- not merely a summary of the main topics covered or a re-statement of your research problem, but a synthesis of key points
- Recap what you did. In about one paragraph recap what your research question was and how you tackled it. Highlight the big accomplishments. Spend another paragraph explaining the highlights of your results. These are the main results you want the reader to remember after they put down the paper, so ignore any small details.
- Finally, finish off with a sentence or two that wraps up your paper. I find this can often be the hardest part to write. You want the paper to feel finished after they read these. One way to do this, is to try and tie your research to the “real world.”

\noindent
**Advantages**  

\noindent
**Disadvantages**  

\noindent
**Limitations**  

\noindent
**Black swan**  

<!-- Make the last 1/2 sentences memorable. -->

## Outlook

\noindent
**More challenging dataset**  

\noindent
**Multiclass classification**  

\noindent
**Tackle crucial challenges**  

## Future work
- Treat cosine similarity as the loss function value for a minimization problem to not just detect a bias, but optimize an existing model using captions.
- Moving window over image while observing the similarity score to locate an object. Or apply segmentation first, then CLIP.
- Find cosine similarities near 0 and 1 (usual range between 0.1 and 0.35)
- "Smilarity" is now defined as sum of large products. 6*6 + 3*3 is the same as 4*4 + 5*5, but they are definitely not similar.
- Suitable to detect highly correlating bias. How about partial bias?
- Time and word limits: How have the limitations of the PhD period restricted your research, or how have the word counts affected the expression of your thesis into a paper?

<!-- The greatest deception men suffer is from their own opinions. Leonardo da Vinci -->

<!-- Future tasks:
- Try alternative datasets:
    - synthetic digits
    - http://ufldl.stanford.edu/housenumbers/
-->

The long-term vision for this XAI method could help removing social bias from machine vision applications and ultimately enhance our lives.
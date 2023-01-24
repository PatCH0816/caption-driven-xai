# Conclusion
<!-- 2-5 sentences, if the explainability method worked as introduction. Then a section with first fat word for each used component. List advantages/disadvantages and limitaions. Last but not least, tell how good/innovative this approach is. -->

## Outlook
<!-- 2 generic introduction sentences. Then list improvements, e.g. more challenging dataset, mutliclass prediction, hyperparameter tuning, etc.
What is the next step? What can one do with this new tech?-->

## Future work
<!-- https://github.com/mlfoundations/open_clip/discussions/361 -->
- Treat cosine similarity as the loss function value for a minimization problem to not just detect a bias, but optimize an existing model using captions.
- Moving window over image while observing the similarity score to locate an object. Or apply segmentation first, then CLIP.
- Find cosine similarities near 0 and 1 (usual range between 0.1 and 0.35)

<!-- The greatest deception men suffer is from their own opinions. Leonardo da Vinci -->

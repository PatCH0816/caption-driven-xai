# Problem description
The advancement in recent years has demonstrated the intriguing performance of several machine learning (ML) models across all kinds of industries. Nevertheless, more than achieving high performance on specific tasks is required. Developers should focus on the robustness of their ML models as well. The idea of robustness is that the ML model masters the desired task and does not fall for a covariate shift between different data distributions. The main focus of this thesis is to develop a caption-based explainable AI method to improve the robustness of ML models in general.

Literature research should provide initial insights into how contrastive language-image pre-training (CLIP) is working at its core. In order to demonstrate the effectiveness of an explainable artificial intelligence (XAI) method to be developed, a suitable task needs to be defined. The dataset for the task of choice should contain a bias. During the training, the ML model should fall for this bias. Using CLIP, it should be demonstrated that the ML model does not learn the chosen task but falls for the bias instead. The objective is to use this knowledge gained with the new XAI method presented to improve the ML model.

\noindent
**Milestones**

- Familiarization with the subject of XAI.
- Literature research on CLIP.
- Create a biased dataset to fool the baseline model.
- Development, training, and analysis of an ML baseline model, which falls for the bias.
- Implementation of a framework using CLIP to explain the ML model.
- Assess the potential of this new XAI method for different use cases.

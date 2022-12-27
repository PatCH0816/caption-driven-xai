# Problem description

\noindent\fbox{
    \begin{minipage}{\linewidth}
        \begin{equation}
            \mathbf{A} \cdot \mathbf{B} = \| \mathbf{A} \| \cdot \| \mathbf{B} \| \cdot cos(\theta)
        \end{equation}
        \begin{equation}
            \sum_{i=1}^{n} \mathbf{A}_i \mathbf{B}_i = \sqrt{ \sum_{i=1}^{n} \mathbf{A}_i^2 } \sqrt{\sum_{i=1}^{n} \mathbf{B}_i^2 } \cdot cos(\theta)
        \end{equation}
        \begin{tabular}{l @{ $=$ } l}
            $\| \mathbf{A} \|$ & Image embedding vector\\
            $\| \mathbf{B} \|$ & Text embedding vector\\
            $ \theta $ & Angle between vectors
        \end{tabular}
    \end{minipage}
}

\noindent\fbox{
    \begin{minipage}{\linewidth}
        \begin{equation}
            cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\| \mathbf{A} \| \cdot \| \mathbf{B} \|} \\
            = \frac{ \sum_{i=1}^{n} \mathbf{A}_i \mathbf{B}_i }{\sqrt{\sum_{i=1}^{n} \mathbf{A}_i^2} \sqrt{\sum_{i=1}^{n} \mathbf{B}_i^2}}
        \end{equation}
        \begin{tabular}{l @{ $=$ } l}
            $\| \mathbf{A} \|$ & Image embedding vector\\
            $\| \mathbf{B} \|$ & Text embedding vector\\
            $\theta$ & Angle between vectors
        \end{tabular}
    \end{minipage}
}

All image and language embedding vectors are normalized. Therefore the shared embedding space of the multi-modal model forms a unit sphere.

![The cosine similarity is a measure of how well two vectors point into the same direction. Since the direction is independent of the length of the vectors, the vectors can be normalized. The effect of the normalization is that all vectors are located on the unit sphere. [[@cosine_similarity]](#references)](source/figures/cosine_similarity.png "Cosine similarity"){#fig:cosine_similarity width=50%}

![A high-level introduction on how CLIP is working. Given an image and some texts, CLIP will asses how well these texts describe the image. (Modified image from    [[@xai_story_bot]](#references))](source/figures/story.png "Story XAI-bot"){#fig:story_xai_bot width=80%}

![The multi-modal CLIP model consists of a text- and an image-encoder. Both encoders produce normalized text- and image-embeddings respectively. Multiplying both embeddings results in a matrix of cosine-similarities. The contrast-learning approach ensures, that the cosine-similarity of the matching image-text-pairs along the diagonal (highlighted in blue) is maximized, while all other products are minimized. This step uses a huge dataset of 400 million text-image-pairs and the process is called contrastive pre-training. [[@clip_process]](#references)](source/figures/clip-1.png "CLIP contrastive pre-training"){#fig:clip-1 width=100%}

![The pre-trained CLIP model has learned an internal representation, which is able to understand a huge number of concepts in text-image-pairs. This model can be used to predict how well a text describes an image from a new dataset of text-image-pairs without the need for any finetuning. This property is called zero-shot prediction capability. [[@clip_process]](#references)](source/figures/clip-2.png "CLIP zero-shot prediction"){#fig:clip-2 width=100%}

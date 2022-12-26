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

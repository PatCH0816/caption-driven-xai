# Contrastive language image pre-training
<!-- what is clip? what does it do? (High level introduction) -->
Contrastive language image pre-training (CLIP) is a multi-modal model, which uses the two modalities language and vision. [@clip_paper] CLIP has been developed by OpenAI as a fundamental element to the success of their famous Dall-E 2 model, which is able to generate images from text prompts. [@dall_e_2_paper] This thesis will not cover details about Dall-E 2, but use CLIP to develop a novel XAI method. On a very high level, the interaction with CLIP can be imagined to look as depicted in \*@fig:story_xai_bot. The objective of CLIP is to assesses how well a specific description matches the context of an image.

![A high-level introduction on how CLIP is working. Given an image and some texts, CLIP will asses how well these texts describe the image. (Modified image from [[@xai_story_bot]](#references))](source/figures/story.png "Story XAI-bot"){#fig:story_xai_bot width=80%}

As shown in \*@fig:clip-1, the structure of CLIP consists of two encoders, one for each modality (Text depicted in violett, vision depicted in green). The text encoder tokenizes the text prompts and uses a transformer to create a text embedding. The image encoder uses a modified ResNet-50 model to create an embedding for the image. This results in high dimensional and normalized image embeddings $\mathbf{I_i} \in \mathbb{R}^{1024}$ and text embedding $\mathbf{T_i} \in \mathbb{R}^{1024}$. The dimensionality of these embeddings are $dim(\mathbf{I_i}) = 1 \times 1024$ and $dim(\mathbf{T_i}) = 1024 \times 1$. This high dimensional embedding space represents all the concepts CLIP knows about. 

![The multi-modal CLIP model consists of a text- and an image-encoder. Both encoders produce normalized text- and image-embeddings respectively. Multiplying both embeddings results in a matrix of cosine-similarities. The contrast-learning approach ensures, that the cosine-similarity of the matching image-text-pairs along the diagonal (highlighted in blue) is maximized, while all other products are minimized. This step uses a huge dataset of 400 million text-image-pairs and the process is called contrastive pre-training. [[@clip_process]](#references)](source/figures/clip-1.png "CLIP contrastive pre-training"){#fig:clip-1 width=100%}

To obtain a score of how well a text prompt describes an image, a measure of similiarity is needed. We are looking to measure the similarity between concepts. E.g. the concept of a tree can be represented by an image of a tree standing on a field of grass under the blue sky and a text prompt "a tree standing on a field of grass under the blue sky". Using both text and image encoders to obtain the text and image embeddings, results in two high dimensional vectors from the shared embedding space. Different directions in this shared embedding space denote different concepts. If two vectors point into the same direction, they point to the same concept and are therefore considered similiar. The angle between two vectors is the property, which denotes, if the vectors are pointing into the same direction. Applying the cosine to this angle $cos(\theta) = [-1, 1]$ results in an theoretical, interpretable range of numbers. The cosine-similarity is derived from the euclidean dot product formula. Note: The text embedding $\mathbf{A}$ and image embedding $\mathbf{B}$ have an arbitrarily magnitude at this point!

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

The cosine-similarity is basically the inner/scalar product of the two normalized text and image embeddings. The shared embedding space of the multi-modal model forms a unit sphere due to the normalization of the text and image embeddings. These normalized text and image embeddings $\frac{\mathbf{A}}{\| \mathbf{A} \|}$ and $\frac{\mathbf{B}}{\| \mathbf{B} \|}$ are denoted as $\mathbf{I_i}$ and text embedding $\mathbf{T_i}$ in figure \*@fig:clip-1.

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

From a mathematical standpoint of view, the theoretical range of numbers for the cosine similarity is $cos(\theta) = [-1, 1]$, for which $cos(\theta) = -1$ denotes opposite concepts, $cos(\theta) = 0$ denotes orthogonal/unrelated concepts and $cos(\theta) = 1$ denotes same concept. As in most non-theoretical physics, there is no concept of negative weight as illustrated in \*@fig:cosine_similarity. The same is true for the text and image embeddings of CLIP. There is either an absent of a concept or a presents of a concept of a different degree, but there are no negative concepts. Therefore, all entries of the text and image embeddings are real numbers equal or larger than 0. This leads to a valid range for the cosine similarity of $cos(\theta) = [0, 1]$.

![This is an image of an hypothetical low-dimensional embedding space to obtain a geometrical understanding of the shared embedding space with limited range. The three axises denote height, weight and length of hair. Using normalized vectors, all persons are located on a unit sphere. The angle between them denotes similarity of these persons in terms of the three properties on the orthogonal axises. Note: There are no negative values for height, weight and length of hair, therefore the valid range of theta is between 0° (Same concept) and 90° (Unrelated concepts). [[@cosine_similarity]](#references)](source/figures/cosine_similarity.png "Cosine similarity"){#fig:cosine_similarity width=50%}

<!-- Explain dataset -->
400 million text-image pairs (With additional data augmentation) have been scraped from the world wide web and used to train the CLIP model by OpenAI. [@clip_paper] They demonstrated that the performance of CLIP is not dependent on their custom dataset by using a CLIP model trained on a subset of their custom dataset and compared the performance of another CLIP model trained on a public available dataset with a similar size. The performance of both models were comparable and therefore OpenAI concluded that the performance of CLIP is not dependend on their custom dataset. Therefore, they decided to keep their custom dataset private and not to release it to the public.

<!-- Explain training -->
To train CLIP, all text and image pairs are passed in minibatches with a size of $N = 32768$ through the text and image encoders, which result in $dim(I) = 32768 \times 1024$ and $dim(\mathbf{T}) = 1024 \times 32768$. These large embeddings lead to a matrix of cosine-similarities of the size $dim(\mathbf{I} \cdot \mathbf{T}) = 32768 \times 32768$. The objective of the training is to maximize the cosine-similarities between the $N = 32768$ correct scores along the diagonal of the matrix. Simultaneously, the scores of the $N^{2} - N = 32768^{2} - 32768$ (All entries of the matrix apart from the diagonal) incorrect pairs need to be minimized. Just to get a feeling for the computational power needed: The training for the largest ResNet Model RN50x64 took 18 day on 592 NVIDIA V100 GPUs.

<!-- Explain zero-shot capability -->
In the usual process of transfer learning, a model is trained on a large dataset and finetuned on a smaller downstream-task specific dataset. The first training on the large dataset is called pre-training. A prediction that is made after finetuning on a number $N$ samples is called a N-shot-prediction. In the case of CLIP, the model is pre-trained by OpenAI. One of the biggest advantages of CLIP is that generally there is no finetuning needed, so a prediction is called a zero-shot-prediction. \*@fig:clip-2 shows how to use CLIP to obtain a zero-shot-prediction. First, one needs to supply text prompts. If there are only class labels available, a template like "a photo of a {object}" can be used, where the {object} will be replaced by the actual class labels. This set of text prompts is fed into the text encoder, which will generate a text embedding. Second, an image needs to be fed into the image encoder, which generates an image embedding. Using the largest cosine-similarity value, the best description for the image can be selected.

![The pre-trained CLIP model has learned an internal representation, which is able to understand a huge number of concepts in text-image-pairs. This model can be used to predict how well a text describes an image from a new dataset of text-image-pairs without the need for any finetuning. This property is called zero-shot prediction capability. [[@clip_process]](#references)](source/figures/clip-2.png "CLIP zero-shot prediction"){#fig:clip-2 width=100%}

Using several text-image-pairs as in \*@fig:clip_matrix, all cosine-similarity scores of the different descriptions and images can be arranged in a confusion matrix. Such a confusion matrix could help to discover a wide set of problems like having ambiguous descriptions.

![In this confusion matrix of CLIP each score indicates how well a description matches the context of an image. (Low scores have a dark color, high scores have a lighter color.)](source/figures/clip_matrix.png "Clip matrix"){#fig:clip_matrix width=100%}

Since this chapter about CLIP is fundamental to the content of this thesis, there is a very brief recap of what the abbreviation of the name stands for:

\noindent
**Contrastive** describes the process of maximizing the cosine-similarities of the correct text-image-pairs (Along the diagonal) and minimizing the cosine-similarity of the incorrect text-image-pairs (All entries of the matrix apart from the diagonal).

\noindent
**Language** denotes the first of the two modalities used in this multi-modal model. Texts belong to the language modality. Text based language is used to suggest matching descriptions for the concepts in images. 

\noindent
**Image** denotes the second of the two modalities used in this multi-modal model. Images belong to the the vision modality. Images are used to find matching text based descriptions.

\noindent
**Pre-training** describes the fact that CLIP has been trained on a large dataset before being used on a downstream-task. There is generally no need to finetine CLIP for a broad spectrum of concepts. Therefore, CLIP possesses zero-shot-prediction capabilities.

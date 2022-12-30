# Contrastive language image pre-training
<!-- what is clip? what does it do? (High level introduction) -->
Contrastive language image pre-training (CLIP) is a multi-modal model which uses the two modalities of language and vision. [@clip_paper] CLIP has been developed by OpenAI as a fundamental element to the success of their famous Dall-E 2 model, which can generate images from text prompts. [@dall_e_2_paper] This thesis will not cover details about Dall-E 2 but will use CLIP to develop a novel XAI method. \*@fig:story_xai_bot illustrates the interaction with CLIP on an abstract level. The objective of CLIP is to assess how well a specific description matches the context of an image.

![This figure demonstrates how to interact with CLIP on an abstract level. Given an image and some texts, CLIP will assess how well these texts describe the image. (Modified image from [[@xai_story_bot]](#references))](source/figures/story.png "Story XAI-bot"){#fig:story_xai_bot width=80%}

As shown in \*@fig:clip-1, the CLIP structure consists of two encoders, one for each modality (Text depicted in purple, vision depicted in green). The text encoder generates tokens and uses a transformer to create an embedding for the text. The image encoder uses a modified ResNet-50 model to create an embedding for the image. This results in high dimensional and normalized image embeddings $\mathbf{I}_i \in \mathbb{R}^{1024}$ and text embedding $\mathbf{T}_i \in \mathbb{R}^{1024}$. The dimensionality of these embeddings are $dim(\mathbf{I}_i) = 1 \times 1024$ and $dim(\mathbf{T}_i) = 1024 \times 1$. This high-dimensional shared embedding space contains all concepts learned by CLIP from the training dataset.

![The multi-modal CLIP model consists of a text and an image encoder. Both encoders produce normalized text and image embeddings, respectively. These two embedding's inner/scalar product results in a matrix of cosine similarities. The contrast-learning approach maximizes the cosine similarity of the matching image-text-pairs along the diagonal (Highlighted in blue) and minimizes the remaining incorrect cosine similarities (Highlighted in grey). This step uses a massive dataset of 400 million text-image pairs and the process is called contrastive pre-training. [[@clip_process]](#references)](source/figures/clip-1.png "CLIP contrastive pre-training"){#fig:clip-1 width=100%}

We are looking to measure the similarity between concepts. A suitable score for this objective should express how well a description fits an image. E.g., the concept of a tree can be represented by an image of a tree standing on a field of grass under the blue sky or a text prompt "a tree standing on a field of grass under the blue sky". Using text and image encoders to obtain the text and image embeddings results in two high-dimensional vectors. Different directions in this shared embedding space denote different concepts. If two vectors point in the same direction, they point to the same concept and are therefore considered similar. The angle between two vectors is the property, which denotes if the vectors are pointing in the same direction. Applying the cosine to this angle $cos(\theta) = [-1, 1]$ results in a theoretical, interpretable range of numbers. The cosine similarity derives from the euclidean dot product formula. Note: The text embedding $\mathbf{A}$ and image embedding $\mathbf{B}$ have an arbitrary magnitude at this point!

\noindent\fbox{
    \begin{minipage}{\linewidth}
        \begin{equation}
            \mathbf{A} \cdot \mathbf{B} = \| \mathbf{A} \| \cdot \| \mathbf{B} \| \cdot cos(\theta)
        \end{equation}
        \begin{equation}
            \sum_{i=1}^{n} \mathbf{A}_i \mathbf{B}_i = \sqrt{ \sum_{i=1}^{n} \mathbf{A}_i^2 } \sqrt{\sum_{i=1}^{n} \mathbf{B}_i^2 } \cdot cos(\theta)
        \end{equation}
        \begin{tabular}{l @{ $=$ } l}
            $\mathbf{A}$ & Image embedding vector\\
            $\mathbf{B}$ & Text embedding vector\\
            $ \theta $ & Angle between vectors
        \end{tabular}
    \end{minipage}
}

The cosine similarity is the inner/scalar product of the two normalized text and image embeddings. The shared embedding space of the multi-modal model forms a unit sphere due to the normalization of the text and image embeddings. These normalized text and image embeddings $\frac{\mathbf{A}}{\| \mathbf{A} \|}$ and $\frac{\mathbf{B}}{\| \mathbf{B} \|}$ are denoted as $\mathbf{I}_i$ and text embedding $\mathbf{T}_i$ in figure \*@fig:clip-1.

\noindent\fbox{
    \begin{minipage}{\linewidth}
        \begin{equation}
            cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\| \mathbf{A} \| \cdot \| \mathbf{B} \|} \\
            = \frac{ \sum_{i=1}^{n} \mathbf{A}_i \mathbf{B}_i }{\sqrt{\sum_{i=1}^{n} \mathbf{A}_i^2} \sqrt{\sum_{i=1}^{n} \mathbf{B}_i^2}}
        \end{equation}
        \begin{tabular}{l @{ $=$ } l}
            $\mathbf{A}$ & Image embedding vector\\
            $\mathbf{B}$ & Text embedding vector\\
            $\theta$ & Angle between vectors
        \end{tabular}
    \end{minipage}
}

From a mathematical standpoint of view, the theoretical range of numbers for the cosine similarity is $cos(\theta) = [-1, 1]$, for which $cos(\theta) = -1$ denotes opposite concepts, $cos(\theta) = 0$ denotes orthogonal/unrelated concepts and $cos(\theta) = 1$ denotes the same concept. As in most non-theoretical physics, there is no concept of negative weight as illustrated in \*@fig:cosine_similarity. The same is true for the text and image embeddings of CLIP. There is either an absence of a concept (There is no tree) or a presence of a concept of a varying degree (There are one or more trees), but there are no negative concepts. (There are no -5 trees) Therefore, all text and image embedding entries are real numbers equal to or larger than 0. This observation leads to a valid range for the cosine similarity of $cos(\theta) = [0, 1]$.

![This illustration shows a hypothetical low-dimensional embedding space to gain a geometrical understanding of the shared embedding space of CLIP. There are three persons Joe, Nancy and Kai of whom we know about their height, weight and length of their hair. All data points are located on a unit sphere in euclidean space using normalized vectors. The angle between them denotes the similarity with respect to the units of the axes. Note: There are no negative values for height, weight or length of hair. Therefore the valid range of theta is between 0° (Same concept) and 90° (Unrelated concepts). [[@cosine_similarity]](#references)](source/figures/cosine_similarity.png "Cosine similarity"){#fig:cosine_similarity width=50%}

<!-- Explain CLIP dataset -->
A suitable dataset was needed to train such a powerful CLIP model. The visual genome dataset is a high-quality crowd-labeled dataset, but with its approximately 100'000 images relatively small by modern standards. Larger datasets like the YFCC100M with 100 million images or the IG-3.5B-17k with 3.5 billion images from Instagram are interesting in terms of the number of images. [@instagram_dataset] However, the associated metadata with these images is sparse and varies in quality. Many images automatically generate useless descriptions with the camera exposure in use or filenames like 20221228_083476.JPG. Therefore, OpenAI created a custom dataset and downloaded 400 million image-test-pairs from the world wide web. To balance this enormous dataset, they used 20,000 image-text pairs per query. [@clip_paper] OpenAI trained CLIP on a subset of their custom dataset and compared the performance of another CLIP model trained on a publicly available dataset with a similar size. The performance of both models was comparable. This result indicates that any well-balanced dataset with a similar size can achieve similar performance compared to OpenAI's custom dataset. Thus, they decided to keep their custom dataset private and not release it to the public.

<!-- Explain open-CLIP and LAION datasets -->
The strong movement in the open source community in recent years continues on large-scale projects like CLIP. The "mlfoundations" community on Github maintains the open-source implementation of CLIP. [@open_clip] The challenge of collecting a massive dataset as OpenAI has been tackled by the Large-scale Artificial Intelligence Open Network (LAION) community. [@laion_datasets] This thesis uses the implementation of OpenAI but wants to take advantage of the opportunity to promote the vital movement of the open-source community.

<!-- Explain training -->
During the training of CLIP, all text and image pairs are passed in mini-batches with a size of $N = 32768$ through the text and image encoders, which result in $dim(\mathbf{I}) = 32768 \times 1024$ and $dim(\mathbf{T}) = 1024 \times 32768$. These large embeddings lead to a matrix of cosine-similarities of the size $dim(\mathbf{I} \cdot \mathbf{T}) = 32768 \times 32768$. The objective of the training is to maximize the cosine similarities of the $N = 32768$ correct scores along the diagonal of the matrix and to minimize the scores of the $N^{2} - N = 32768^{2} - 32768$ (All entries of the matrix apart from the diagonal) incorrect pairs. To get a feeling for the computational power needed: The training for the largest ResNet Model RN50x64 took 18 days on 592 NVIDIA V100 GPUs.

<!-- Explain zero-shot capability -->
In the usual transfer learning process, a model is trained on a large dataset (This is called pre-training) and finetuned on a smaller downstream-task-specific dataset. After finetuning the model on a number $N$ samples, the prediction is called an N-shot prediction. In the case of CLIP, the model is pre-trained by OpenAI. One of the most significant advantages of CLIP is that generally, there is no finetuning needed, so a prediction is called a zero-shot-prediction. \*@fig:clip-2 shows how to use CLIP to obtain a zero-shot-prediction. First, one needs to supply text prompts. Using class labels, a template like "a photo of an {object}" can be used, where the actual class labels can replace the {object}. This text prompt set is fed into the text encoder, generating a text embedding. Feeding an image into the image encoder generates an image embedding. The best description for the image is the one with the most significant cosine similarity score.

![The pre-trained CLIP model has learned an internal representation, which can understand many concepts in text-image-pairs. This model predicts how well a text describes an image from a new dataset of text-image-pairs without any finetuning. This property is called zero-shot prediction capability. [[@clip_process]](#references)](source/figures/clip-2.png "CLIP zero-shot prediction"){#fig:clip-2 width=100%}

Using several text-image-pairs as in \*@fig:clip_matrix, all cosine-similarity scores of the different descriptions and images can be arranged in a confusion matrix. Such a confusion matrix could help to discover a broad set of problems, like ambiguous descriptions.

![In this confusion matrix of CLIP, each score indicates how well a description matches the context of an image. (Low scores have a dark color whether high scores have a lighter color)](source/figures/clip_matrix.png "Clip matrix"){#fig:clip_matrix width=100%}

Since this chapter about CLIP is fundamental to the content of this thesis, there is a very brief recap of what the abbreviation of the name stands for:

\noindent
**Contrastive** describes the process of maximizing the cosine-similarities of the correct text-image-pairs (Along the diagonal) and minimizing the cosine-similarity of the incorrect text-image-pairs (All entries of the matrix apart from the diagonal).

\noindent
**Language** denotes the first of the two modalities used in this multi-modal model. Texts belong to the language modality. Text-based language is used to suggest matching descriptions for the concepts in images. 

\noindent
**Image** denotes the second of the two modalities used in this multi-modal model. Images belong to the vision modality. Images are used to find matching text-based descriptions.

\noindent
**Pre-training** describes that CLIP has been trained on a large dataset before being used on a downstream task. There is generally no need to finetune CLIP for a broad spectrum of concepts. Therefore, CLIP possesses zero-shot-prediction capabilities.

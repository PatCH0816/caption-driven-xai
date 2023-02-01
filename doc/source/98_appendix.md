# Appendix
## Configurations of CLIP
OpenAI's CLIP architecture is a successful demonstration of the power of a self-supervised model to learn a wide variety of concepts from caption-image-pairs. During the course of the research, OpenAI has used different variations for their image encoder, which is summarized in the \*@tbl:clip_configuration_table.

| Image encoder     | Text encoder                      | Dataset
|-                  | -                                 | -
| RN50              | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| RN101             | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| RN50x4            | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| RN50x16           | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| RN50x64           | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| ViT-B/32          | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| ViT-B/16          | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| ViT-L/14          | Masked self-attention Transformer | Proprietary OpenAI custom dataset
| ViT-L/14@336px    | Masked self-attention Transformer | Proprietary OpenAI custom dataset
Table: A summary of all released versions of CLIP from OpenAI. {#tbl:clip_configuration_table}

The opensource movement noticed the huge attention that CLIP received and deserved. Therefore, the community around mlfoundations on Github successfully deployed a python package, which provides their own versions of CLIP trained on public datasets like LAION and OpenAI's version of CLIP. A short overview is summarized in the \*@tbl:open_clip_configuration_table.

| Image encoder         | Text encoder                       | Dataset
|-                      | -                                  | -
| RN50                  | Masked self-attention Transformer  | openai
| RN50                  | Masked self-attention Transformer  | yfcc15m
| RN50                  | Masked self-attention Transformer  | cc12m
| RN50-quickgelu        | Masked self-attention Transformer  | openai
| RN50-quickgelu        | Masked self-attention Transformer  | yfcc15m
| RN50-quickgelu        | Masked self-attention Transformer  | cc12m
| RN101                 | Masked self-attention Transformer  | openai
| RN101                 | Masked self-attention Transformer  | yfcc15m
| RN101-quickgelu       | Masked self-attention Transformer  | openai
| RN101-quickgelu       | Masked self-attention Transformer  | yfcc15m
| RN50x4                | Masked self-attention Transformer  | openai
| RN50x16               | Masked self-attention Transformer  | openai
| RN50x64               | Masked self-attention Transformer  | openai
| ViT-B-32              | Masked self-attention Transformer  | openai
| ViT-B-32              | Masked self-attention Transformer  | laion400m_e31
| ViT-B-32              | Masked self-attention Transformer  | laion400m_e32
| ViT-B-32              | Masked self-attention Transformer  | laion2b_e16
| ViT-B-32              | Masked self-attention Transformer  | laion2b_s34b_b79k
| ViT-B-32-quickgelu    | Masked self-attention Transformer  | openai
| ViT-B-32-quickgelu    | Masked self-attention Transformer  | laion400m_e31
| ViT-B-32-quickgelu    | Masked self-attention Transformer  | laion400m_e32
| ViT-B-16              | Masked self-attention Transformer  | openai
| ViT-B-16              | Masked self-attention Transformer  | laion400m_e31
| ViT-B-16              | Masked self-attention Transformer  | laion400m_e32
| ViT-B-16-plus-240     | Masked self-attention Transformer  | laion400m_e31
| ViT-B-16-plus-240     | Masked self-attention Transformer  | laion400m_e32
| ViT-L-14              | Masked self-attention Transformer  | openai
| ViT-L-14              | Masked self-attention Transformer  | laion400m_e31
| ViT-L-14              | Masked self-attention Transformer  | laion400m_e32
| ViT-L-14              | Masked self-attention Transformer  | laion2b_s32b_b82k
| ViT-L-14-336          | Masked self-attention Transformer  | openai
| ViT-H-14              | Masked self-attention Transformer  | laion2b_s32b_b79k
| ViT-g-14              | Masked self-attention Transformer  | laion2b_s12b_b42k]
Table: A summary of all released versions of CLIP from the opensource community mlfoundations on Github. {#tbl:open_clip_configuration_table}

## Differences between CLIP and open-CLIP
It is understandable that open-CLIP can be trained on public datasets like LAION as listed in \*@tbl:open_clip_configuration_table. To simplify the process of comparing the results between CLIP and open-CLIP, the open-source community provides the original implementations trained on the OpenAI datasets as well. If one wants to compare the two models (Which should be exactly the same, because the OpenAI dataset is proprietary), one will find that the text-preprocessor and the image encoder are exactly the same in CLIP and open-CLIP. This is expected, however, the architecture of the text-transformer differs. To compare e.g. the architecture for the two ResNet-50 (RN50) image encoders, one has to import CLIP like this:

```python
import clip

model, preprocess = clip.load("RN50")
model
```

The CLIP implementation by open-CLIP can be imported like this:

```python
import open_clip

open_model, _, open_preprocess = \
    open_clip.create_model_and_transforms('RN50',
                                          pretrained='openai')
open_model
```
Using these two import code snippets, a developer expects to obtain the same CLIP architecture trained on the proprietary OpenAI dataset. However, the text-transformer architecture contains some differences, as shown in \*@fig:diff_clip_vs_open_clip. The author started an open discussion on Github about these minor numeric differences. [@difference_clip_vs_open_clip] The changes in the architecture are non-functional and do not influence the numeric results at all. The reason for the minor differences is that open-CLIP uses torch.float32 datatypes in all layers, while the original CLIP implementation uses a mixture between torch.float32 and torch.float16 datatypes.

![Different architectures for the CLIP (Left) and open-CLIP (Right) implementations for the text-transformers for the same argument provided to load CLIP with a ResNet-50 image encoder.](source/figures/diff_clip_vs_open_clip.png "Different architectures in CLIP and open-CLIP"){#fig:diff_clip_vs_open_clip width=100%}

## Embedding dimensions of CLIP
Each caption-image encoder configuration of CLIP in \*@tbl:open_clip_configuration_table generates an embedding with a distinctive dimensionality summarized in \*@tbl:clip_embedding_dimensions. The dimensionality of the image and text encoder are identical.

| Image encoder         | Embedding dimensions
|-                      | -
| RN50                  | 1024
| RN101                 | 512
| RN50x4                | 640
| RN50x16               | 768
| RN50x64               | 1024
| ViT-B/32              | 512
| ViT-B/16              | 512
| ViT-L/14              | 768
| ViT-L/14@336px        | 768
Table: The different embedding dimensionalities of all available CLIP configurations. {#tbl:clip_embedding_dimensions}

## CLIP preprocessor
Every CLIP image encoder has its preprocessor. \*@tbl:clip_rn50_preprocess summarizes the configuration for the associated preprocessor of the ResNet-50 image encoder.

| Operation                                                                 | Explanation
|-                                                                          | -
| Resize(size=224, interpolation=bicubic, max_size=None, antialias=None)    | Resize the image such that the shorter edge of the image is 224 pixels.
| CenterCrop(size=(224, 224))                                               | Crop 224x224 pixels from the center to ensure a quadratic format.
| <function _convert_image_to_rgb at <Hexadecimal-address>>                 | Transform 1x224x224 grayscale images to 3x224x224
| ToTensor()                                                                | Ensure tensor format.
| Normalize(mean=(..), std=(..))                                            | Standard scale pixel values.
Table: Overview of chronological operations from the preprocessor from CLIP's ResNet-50 image encoder according to explanations. {#tbl:clip_rn50_preprocess}

## CLIP image encoder
<!-- CLIP paper chapter: Choosing and Scaling a Model -->
As documented in \*@sec:configurations-of-clip, many different configurations for CLIP are available. In this work, a ResNet-50 model is used as the image encoder due to its proven performance and widespread adoption. OpenAi incorporates several modifications to the "base" ResNet-50 model as described below:

- Instead of either increasing the width of the model [@mahajan_limits_weakly_supervised_pretraining] or the depth [@he_deep_residual_learning] only, use the approach of increasing the width, depth and resolution of the model simultaneously. [@tan_efficientnet]
- ResNet-50-D improves ResNet-B by adding a 2x2 average pooling layer with a stride of 2 before the 1x1 convolutions. This helps not to ignore $\frac{3}{4}$ of activation maps in 1x1 convolutions. [@he_bag_of_tricks]
- Add antialiased rect-2 blur pooling to improve the model shift-invariant characteristic. [@zhang_cnn_shift_invariant]
- Use attention pooling instead of the global average pooling layer. [@global_average_pooling] The attention pooling is implemented as a single layer of "transformer-style" multi-head QKV attention where the query is conditioned on the global average-pooled representation of the image. [@clip_paper]

## CLIP text encoder
<!-- CLIP paper chapter: Choosing and Scaling a Model -->
As documented in \*@sec:configurations-of-clip, many different configurations for CLIP are available. In this work, a transformer is used as the text encoder. OpenAi incorporates several modifications to the "base" transformer as described below:

- Based on transformer architecture with 63M-parameter 12- layer 512-wide model with eight attention heads from Vaswani et al. [@vaswani_attention_is_all_you_need]
- Architecture modifications described in Radford et al., like moving the layer normalization to the input of each sub-block. After the final self-attention block, an additional normalization layer was added. A novel initialization method, which considers the depth of the residual path, is used with a factor $\frac{1}{\sqrt{N}}$. (N is the number of residual layers.) The context size has been doubled from 512 to 1024 and the batch size has been increased to 512. [@radford_language_models]
- The transformer learned a vocabulary size of 49'152 and operated on lower-case byte pair encoding representation. [@sennrich_rare_words_with_subword_units]
- The maximum length of a sequence is limited to 76 for computational efficiency.
- The start of a text sequence is bracketed with [SOS] and ends with [EOS]. The activations of the highest layer of the transformer at the [EOS] token are treated as the feature representation of the text, which is layer normalized and then linearly projected into the multi-modal embedding space.
- To preserve the ability to add language modeling or initialize with a pre-trained language model, masked self-attention is used in the text encoder.
- CLIP's performance is less sensitive to the capacity of the text encoder. Therefore, only the width of the model is scaled to match the width of the ResNet image encoder proportionally. [@clip_paper]

## Additional layer swapping results standalone model
<!-- 
We don't care about how CLIP worked before swapping, because this is arbitrarily. The "after swapping" scores are also dependent on the inital CLIP performance. Only the relative differences describe, how the accuracies of the captions changed because of the layer swapping. 
-->
\*@tbl:biased_training_results shows the caption-based explainable AI model's results incorporating the standalone model using network surgery evaluated on the training dataset. The caption-based explainable AI model successfully reveals the color as the dominant concept of the standalone model as shown in \*@fig:dominant_concept_training_1. \*@fig:dominant_concept_training_2 shows a different representation of the same observation with grouped results by their concept of color and shape.

<!-- | Biased training  | Normalize(cos_sim(No layer swapped))      | Normalize(cos_sim(four layer swapped))| Normalize(cos_sim(four layer swapped - No layer swapped)) -->
<!-- | Biased training  | Before swapping (Original CLIP accuracy)  | After swapping (Absolute difference)  | After swapping (Relative difference) -->
| Biased training       | Zero layers swapped                       | Four layers swapped                   | Influence network surgery
|-                      | -                                         | -                                     | -
|Correct shape          | 0.00%                                     | 0.00%                                 | 11.83%
|Correct color          | 49.60%                                    | 51.37%                                | 38.75%
|Wrong shape            | 0.00%                                     | 0.00%                                 | 12.26%
|Wrong color            | 50.40%                                    | 48.63%                                | 37.16%
Table: Results of the caption-based explainable AI model using the biased standalone model and the training dataset. {#tbl:biased_training_results}

![The color is the dominant concept for the standalone model evaluated on the training dataset, as revealed by the caption-based explainable AI model. The results display the normalized number of corrects/wrong predictions per concept.](source/figures/dominant_concept_training_1.png "The color is the dominant concept for the standalone model evaluated on the training dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_training_1 width=75%}

![The color is the dominant concept for the standalone model evaluated on the training dataset, as revealed by the caption-based explainable AI model. The results display the relative number of predictions per concept.](source/figures/dominant_concept_training_2.png "The color is the dominant concept for the standalone model evaluated on the training dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_training_2 width=75%}

\*@tbl:biased_real_world_results shows the caption-based explainable AI model's results incorporating the standalone model using network surgery evaluated on the real-world dataset. The caption-based explainable AI model successfully reveals the color as the dominant concept of the standalone model as shown in \*@fig:dominant_concept_real_world_1. \*@fig:dominant_concept_real_world_2 shows a different representation of the same observation with grouped results by their concept of color and shape.

| Biased real-world     | Zero layers swapped                       | Four layers swapped                   | Influence network surgery
|-                      | -                                         | -                                     | -
|Correct shape          | 0.05%                                     | 0.00%                                 | 10.55%
|Correct color          | 48.00%                                    | 52.45%                                | 41.10%
|Wrong shape            | 0.00%                                     | 0.00%                                 | 11.80%
|Wrong color            | 51.95%                                    | 47.55%                                | 36.55%
Table: Results of the caption-based explainable AI model using the biased standalone model and the real-world dataset. {#tbl:biased_real_world_results}

![The color is the dominant concept for the standalone model evaluated on the real-world dataset, as revealed by the caption-based explainable AI model. The results display the normalized number of corrects/wrong predictions per concept.](source/figures/dominant_concept_real_world_1.png "The color is the dominant concept for the standalone model evaluated on the real-world dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_real_world_1 width=75%}

![The color is the dominant concept for the standalone model evaluated on the real-world dataset, as revealed by the caption-based explainable AI model. The results display the relative number of predictions per concept.](source/figures/dominant_concept_real_world_2.png "The color is the dominant concept for the standalone model evaluated on the real-world dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_real_world_2 width=75%}

## Additional layer swapping results unbiased standalone model
\*@tbl:unbiased_training_results shows the caption-based explainable AI model's results incorporating the unbiased standalone model using network surgery evaluated on the training dataset. In order to identify the dominant concept, the changes between the results in \*@sec:additional-layer-swapping-results-standalone-model and the results in this section need to be analyzed. The values regarding the shapes of the column "Influence network surgery" in this section have risen considerably, while the sum of the colors has decreased. The caption-based explainable AI model successfully reveals the shape as the dominant concept of the unbiased standalone model as shown in \*@fig:dominant_concept_training_3. \*@fig:dominant_concept_training_4 shows a different representation of the same observation with grouped results by their concept of color and shape.

| Unbiased training     | Zero layers swapped                       | Four layers swapped                   | Influence network surgery
|-                      | -                                         | -                                     | -
|Correct shape          | 36.21%                                    | 37.21%                                | 33.91%
|Wrong shape            | 37.87%                                    | 34.57%                                | 32.06%
|Any color              | 25.92%                                    | 28.22%                                | 34.03%
Table: Results of the caption-based explainable AI model using the unbiased standalone model and the training dataset. {#tbl:unbiased_training_results}

![The shape is the dominant concept for the unbiased standalone model evaluated on the training dataset, as revealed by the caption-based explainable AI model. The results display the normalized number of corrects/wrong predictions per concept.](source/figures/dominant_concept_training_3.png "The color is the dominant concept for the unbiased standalone model evaluated on the training dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_training_3 width=75%}

![The shape is the dominant concept for the unbiased standalone model evaluated on the training dataset, as revealed by the caption-based explainable AI model. The results display the relative number of predictions per concept.](source/figures/dominant_concept_training_4.png "The color is the dominant concept for the unbiased standalone model evaluated on the training dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_training_4 width=75%}

\*@tbl:unbiased_real_world_results shows the caption-based explainable AI model's results incorporating the unbiased standalone model using network surgery evaluated on the real-world dataset. In order to identify the dominant concept, the changes between the results in \*@sec:additional-layer-swapping-results-standalone-model and the results in this section need to be analyzed. The values regarding the shapes of the column "Influence network surgery" in this section have risen considerably, while the sum of the colors has decreased. The caption-based explainable AI model successfully reveals the shape as the dominant concept of the unbiased standalone model, as shown in \*@fig:dominant_concept_real_world_3. \*@fig:dominant_concept_real_world_4 shows a different representation of the same observation with grouped results by their concept of color and shape.

| Unbiased real-world   | Zero layers swapped                       | Four layers swapped                   | Influence network surgery
|-                      | -                                         | -                                     | -
|Correct shape          | 40.35%                                    | 39.55%                                | 35.35%
|Wrong shape            | 38.50%                                    | 36.10%                                | 34.20%
|Any color              | 21.15%                                    | 24.35%                                | 30.45%
Table: Results of the caption-based explainable AI model using the unbiased standalone model and the real-world dataset. {#tbl:unbiased_real_world_results}

![The shape is the dominant concept for the unbiased standalone model evaluated on the training dataset, as revealed by the caption-based explainable AI model. The results display the normalized number of corrects/wrong predictions per concept.](source/figures/dominant_concept_real_world_3.png "The color is the dominant concept for the unbiased standalone model evaluated on the training dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_real_world_3 width=75%}

![The shape is the dominant concept for the unbiased standalone model evaluated on the training dataset, as revealed by the caption-based explainable AI model. The results display the relative number of predictions per concept.](source/figures/dominant_concept_real_world_4.png "The color is the dominant concept for the unbiased standalone model evaluated on the training dataset as revealed by the caption-based explainable AI model."){#fig:dominant_concept_real_world_4 width=75%}

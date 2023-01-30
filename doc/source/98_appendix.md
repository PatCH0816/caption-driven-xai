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

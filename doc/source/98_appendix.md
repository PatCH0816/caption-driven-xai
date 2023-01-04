# Appendix
## Configurations of CLIP
OpenAI's CLIP architecture is a successful demonstration of the power of a self-supervised model to learn a wide variety of concepts from text-image-pairs. During the course of the research, OpenAI has used different variations for their image encoder, which is summarized in the \*@tbl:clip_configuration_table.

| Image encoder     | Text encoder          | Dataset
|-                  | -                     | -
| RN50              | Modified transformer  | Proprietary OpenAI custom dataset
| RN101             | Modified transformer  | Proprietary OpenAI custom dataset
| RN50x4            | Modified transformer  | Proprietary OpenAI custom dataset
| RN50x16           | Modified transformer  | Proprietary OpenAI custom dataset
| RN50x64           | Modified transformer  | Proprietary OpenAI custom dataset
| ViT-B/32          | Modified transformer  | Proprietary OpenAI custom dataset
| ViT-B/16          | Modified transformer  | Proprietary OpenAI custom dataset
| ViT-L/14          | Modified transformer  | Proprietary OpenAI custom dataset
| ViT-L/14@336px    | Modified transformer  | Proprietary OpenAI custom dataset
Table: A summary of all released versions of CLIP from OpenAI. {#tbl:clip_configuration_table}

The opensource movement noticed the huge attention that CLIP received and deserved. Therefore, the community around mlfoundations on Github successfully deployed a python package, which provides their own versions of CLIP trained on public datasets like LAION and OpenAI's version of CLIP. A short overview is summarized in the \*@tbl:open_clip_configuration_table.

| Image encoder         | Text encoder          | Dataset
|-                      | -                     | -
| RN50                  | Modified transformer  | openai
| RN50                  | Modified transformer  | yfcc15m
| RN50                  | Modified transformer  | cc12m
| RN50-quickgelu        | Modified transformer  | openai
| RN50-quickgelu        | Modified transformer  | yfcc15m
| RN50-quickgelu        | Modified transformer  | cc12m
| RN101                 | Modified transformer  | openai
| RN101                 | Modified transformer  | yfcc15m
| RN101-quickgelu       | Modified transformer  | openai
| RN101-quickgelu       | Modified transformer  | yfcc15m
| RN50x4                | Modified transformer  | openai
| RN50x16               | Modified transformer  | openai
| RN50x64               | Modified transformer  | openai
| ViT-B-32              | Modified transformer  | openai
| ViT-B-32              | Modified transformer  | laion400m_e31
| ViT-B-32              | Modified transformer  | laion400m_e32
| ViT-B-32              | Modified transformer  | laion2b_e16
| ViT-B-32              | Modified transformer  | laion2b_s34b_b79k
| ViT-B-32-quickgelu    | Modified transformer  | openai
| ViT-B-32-quickgelu    | Modified transformer  | laion400m_e31
| ViT-B-32-quickgelu    | Modified transformer  | laion400m_e32
| ViT-B-16              | Modified transformer  | openai
| ViT-B-16              | Modified transformer  | laion400m_e31
| ViT-B-16              | Modified transformer  | laion400m_e32
| ViT-B-16-plus-240     | Modified transformer  | laion400m_e31
| ViT-B-16-plus-240     | Modified transformer  | laion400m_e32
| ViT-L-14              | Modified transformer  | openai
| ViT-L-14              | Modified transformer  | laion400m_e31
| ViT-L-14              | Modified transformer  | laion400m_e32
| ViT-L-14              | Modified transformer  | laion2b_s32b_b82k
| ViT-L-14-336          | Modified transformer  | openai
| ViT-H-14              | Modified transformer  | laion2b_s32b_b79k
| ViT-g-14              | Modified transformer  | laion2b_s12b_b42k]
Table: A summary of all released versions of CLIP from the opensource community mlfoundations on Github. {#tbl:open_clip_configuration_table}

## Differences between CLIP and open-CLIP
It is understandable that open-CLIP can be trained on public datasets like LAION as listed in \*@tbl:open_clip_configuration_table. To simplify the process of comparing the results between CLIP and open-CLIP, the open-source community provides the original implementations trained on the OpenAI datasets as well. If one wants to compare the two models (Which should be exactly the same, because the OpenAI dataset is proprietary), one will find that the text-preprocessor and the image-encoder are exactly the same in CLIP and open-CLIP. This is expected, however, the architecture of the text-transformer differs. To compare e.g. the architecture for the two ResNet-50 (RN50) image-encoders, one has to import CLIP like this:

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

Using these two import code snippets, a developer expects to obtain the exact same CLIP-architecture trained on the proprietary OpenAI dataset. However, the text-transformer architecture contains some differences as shown in \*@fig:diff_clip_vs_open_clip. Even if the difference in the architecture of the text-transformer can be justified, because the loading functions receive a specific configuration for the image encoder (RN50) only, it is not clear how the new text-transformer model has been trained. The problem is that the OpenAI custom dataset is proprietary. This leads to unexpected small differences in the results between CLIP and open-CLIP. The author started an open discussion on Github. [@difference_clip_vs_open_clip]

![Different architectures for the CLIP (Left) and open-CLIP (Right) implementations for the text-transformers for the same argument provided to load CLIP with a ResNet-50 image encoder.](source/figures/diff_clip_vs_open_clip.png "Different architectures in CLIP and open-CLIP"){#fig:diff_clip_vs_open_clip width=100%}

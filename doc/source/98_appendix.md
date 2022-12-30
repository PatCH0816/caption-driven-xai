# Appendix
## Configurations of CLIP
OpenAI's CLIP architecture is a successful demonstration of the power of a self-supervised model to learn a wide variety of concepts from text-image-pairs. During the course of the research, OpenAI has used different variations for their image encoder, which is summarized in the \*@tbl:clip_configuration_table.

| Image encoder     | Text encoder          | Dataset
|-                  | -                     | -
| RN50              | Modified transformer  | Proprietary custom dataset
| RN101             | Modified transformer  | Proprietary custom dataset
| RN50x4            | Modified transformer  | Proprietary custom dataset
| RN50x16           | Modified transformer  | Proprietary custom dataset
| RN50x64           | Modified transformer  | Proprietary custom dataset
| ViT-B/32          | Modified transformer  | Proprietary custom dataset
| ViT-B/16          | Modified transformer  | Proprietary custom dataset
| ViT-L/14          | Modified transformer  | Proprietary custom dataset
| ViT-L/14@336px    | Modified transformer  | Proprietary custom dataset
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

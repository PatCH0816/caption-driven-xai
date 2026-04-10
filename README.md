# 🧠 Caption-Driven Explainability: Probing CNNs for Bias via CLIP

[![DOI](https://zenodo.org/badge/541802929.svg)](https://zenodo.org/badge/latestdoi/541802929)
[![arXiv](https://img.shields.io/badge/arXiv-2510.22035-b31b1b)](https://arxiv.org/abs/2510.22035)
[![ICIP 2025](https://img.shields.io/badge/ICIP-2025-blue)](https://ieeexplore.ieee.org/document/11386015)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Google Scholar](https://img.shields.io/badge/Google%20Scholar-Publications-blue?logo=google-scholar)](https://scholar.google.com/citations?user=jMiy9HQAAAAJ&hl=en)

> Multimodal explainable AI framework combining CLIP and CNNs to reveal concept-level bias and interpretability in deep vision models.

> **Official implementation of the paper:**  
> [Patrick Koller](https://scholar.google.com/citations?user=jMiy9HQAAAAJ&hl=en)¹  
> [Amil Dravid](https://scholar.google.com/citations?user=YZ8Y-sUAAAAJ)² *(also published as Amil V. Dravid)*  
> [Guido M. Schuster](https://scholar.google.com/citations?user=_A1-3vMAAAAJ&hl=de&oi=ao)³  
> [Aggelos K. Katsaggelos](https://scholar.google.com/citations?user=aucB85kAAAAJ&hl=en)¹  
>
> ¹Northwestern University | ²UC Berkeley | ³Eastern Switzerland University of Applied Sciences  
> 🏔️ Presented at **IEEE ICIP 2025**, Anchorage (Alaska)
<!-- Alternate author naming: Amil Dravid (Google Scholar), Amil V. Dravid (arXiv/IEEE) -->

---

## 🚀 Overview

Deep neural networks have transformed computer vision, achieving remarkable accuracy in recognition, detection, and classification tasks.  
However, understanding *why* a network makes a specific decision remains one of the central challenges in AI.  
This repository introduces a **multimodal explainable AI (XAI)** framework that bridges **vision and language** using **OpenAI's CLIP**.  
Through a process called **network surgery**, it reveals the semantic concepts driving model predictions and exposes hidden **biases** within learned representations.

💡 Unlike pixel-based saliency methods, our approach:
- Explains *what concept* drives a prediction, not just *where* the model looked  
- Identifies **spurious correlations** such as color or texture bias  
- Provides **quantitative insight** into robustness and covariate shift  

<p align="center">
  <img src="doc/source/figures/abstract/abstract_3_xai.png" width="700"><br>
  <em>Conceptual overview: bridging CLIP and a standalone model to uncover the semantics behind decisions.</em>
</p>

**This repository contains:**
- ✅ Full inference pipeline for caption-driven XAI
- ✅ CLIP-based probing utilities
- ✅ Network surgery implementation
- ✅ Bias visualization assets
- ✅ Example datasets & scripts


---

## 🧩 Core Idea

We integrate a **standalone model to be explained** (for example ResNet-50) into **CLIP** by aligning their activation maps.  
CLIP’s text encoder then serves as a semantic probe, describing *what* the model has truly learned.

### 🔍 Key Components
1. **Network surgery** – Swap correlated activation maps between the standalone model and CLIP  
2. **Activation matching** – Compute cross-layer correlations to identify equivalent feature spaces  
3. **Caption-based inference** – Use natural-language captions (e.g. “red digit”, “green digit”, “round shape”) to interpret dominant concepts  

<p align="center">
  <img src="doc/source/figures/abstract/abstract_2_clip.png" width="650"><br>
  <em>Activation matching aligns internal feature spaces for interpretable concept fusion.</em>
</p>

---

## ⚖️ Grad-CAM vs. Caption-Driven XAI

Both **Grad-CAM** and **Caption-Driven XAI** offer valuable insights, but they answer different questions.

| Method | Explains | Handles overlapping features | Quantitative concept analysis | Human-readable output |
|:--|:--|:--|:--|:--|
| **Grad-CAM** | Spatial importance (*where*) | ❌ | ❌ | ❌ |
| **Caption-Driven XAI** | Conceptual semantics (*what*) | ✅ | ✅ | ✅ |

Grad-CAM highlights the *region* of attention, while Caption-Driven XAI uncovers the *reason*, bridging visual focus with linguistic meaning.  
*Quantitative concept analysis refers to measuring how strongly each linguistic concept (e.g. “red”, “round”) influences a model’s prediction, based on similarity in CLIP’s multimodal embedding space.*

---

## 📚 Citation

If you use this repository, please cite:

```bibtex
@inproceedings{koller2025captionxai,
  title={Caption-Driven Explainability: Probing CNNs for Bias via CLIP},
  author={Koller, Patrick and Dravid, Amil V. and Schuster, Guido M. and Katsaggelos, Aggelos K.},
  booktitle={IEEE International Conference on Image Processing (ICIP) – Satellite Workshop on Generative AI for World Simulations and Communications},
  year={2025},
  note={Preprint available at arXiv:2510.22035}
}
```


## 🌐 Links

- 📄 **arXiv preprint:** https://arxiv.org/abs/2510.22035  
- 🧪 **Zenodo archive (v1.0.0):** https://doi.org/10.5281/zenodo.17546054  
- 👤 **Personal website:** https://patch0816.github.io  
- 🎓 **Google Scholar:** https://scholar.google.com/citations?user=jMiy9HQAAAAJ&hl=en



## ❤️ Acknowledgments

This research was conducted at the [AIM-IVPL Lab](https://sites.northwestern.edu/ivpl/) (Northwestern University),  
in collaboration with [UC Berkeley](https://www.berkeley.edu/) and [OST/ICAI Switzerland](https://www.ost.ch/de/forschung-und-dienstleistungen/interdisziplinaere-themen/icai-interdisciplinary-center-for-artificial-intelligence).

---
*Keywords:* Explainable AI, CLIP, Computer Vision, Bias, Robustness, Interpretability, Multimodal Learning, Northwestern University, ICIP 2025
---

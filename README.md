# dino-attention-heatmaps

Generate **paper-style patch-grid visualizations** from Vision Transformers:
- **Attention patchmap**: [CLS] → patch attention (Softmax probabilities)
- **PCA patchmap**: RGB visualization of patch token embeddings (DINO-style “AI visuals”)

Supports:
- **DINO** (torch.hub)
- **DINOv2** (torch.hub)
- **DINOv3** (Hugging Face Transformers)

---

## Why this repo?

Many DINO/DINOv2/DINOv3 figures show patch-level visualizations such as:
- multi-head self-attention (MHSA) maps
- patch token PCA maps

This repo provides a simple CLI to reproduce these visualizations from any input image, with a resolution option.

---

## Outputs

### 1) Attention patchmap (CLS → patches)
A patch-grid heatmap where each square corresponds to one ViT patch token, colored by the **attention probability** from the **[CLS] token** to that patch.

### 2) PCA patchmap (RGB tokens)
A patch-grid RGB visualization from the first 3 PCA components of patch token embeddings.

---

## Installation

### Option A: Conda (recommended)

```bash
conda env create -f environment.yml
conda activate dino

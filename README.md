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
````

### Option B: pip

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

---

## Quickstart

### DINOv2 — attention patchmap

```bash
python attention_heatmap.py \
  --image examples/input.jpg \
  --model_type dinov2 \
  --model dinov2_vitb14 \
  --res 224 \
  --viz attention \
  --interp nearest \
  --out_prefix examples/out
```

### DINOv2 — PCA patchmap (DINO-style “AI visuals”)

```bash
python attention_heatmap.py \
  --image examples/input.jpg \
  --model_type dinov2 \
  --model dinov2_vitb14 \
  --res 224 \
  --viz pca \
  --interp nearest \
  --out_prefix examples/out
```

### Higher resolution (e.g., 896×896)

```bash
python attention_heatmap.py \
  --image examples/input.jpg \
  --model_type dinov2 \
  --model dinov2_vitb14 \
  --res 896 \
  --viz both \
  --interp nearest \
  --out_prefix examples/out
```

### DINOv3 (Hugging Face)

```bash
python attention_heatmap.py \
  --image examples/input.jpg \
  --model_type dinov3 \
  --model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --res 896 \
  --viz both \
  --interp nearest \
  --out_prefix examples/out
```

---

## CLI Options

| Arg            | Meaning                                                |
| -------------- | ------------------------------------------------------ |
| `--image`      | Path to input image                                    |
| `--model_type` | `dino`, `dinov2`, or `dinov3`                          |
| `--model`      | Hub model name (dino/dinov2) or HF model id (dinov3)   |
| `--res`        | Square resolution (e.g., 224, 448, 896, 1344)          |
| `--viz`        | `attention`, `pca`, or `both`                          |
| `--cmap`       | Colormap for attention patchmap (default `viridis`)    |
| `--interp`     | Patch upsampling: `nearest` (paper-like) or `bilinear` |
| `--out_prefix` | Prefix for output filenames                            |
| `--device`     | `cuda` or `cpu`                                        |

---

## Notes

* The attention map is computed from the **last transformer block** as:

  * compute Q,K,V
  * apply scaled dot-product attention
  * apply **Softmax**
  * average over heads
  * extract **[CLS] → patch** attentions
* PCA patchmaps use patch token embeddings and project to RGB with PCA.

---

## Troubleshooting

### "xFormers is not available"

This is normal on many systems (especially macOS). The script will still work.

### Torch hub download is slow

The first run downloads weights into:
`~/.cache/torch/hub/` and `~/.cache/torch/hub/checkpoints/`

---

## Citation / Credit

This repo uses publicly available pretrained models:

* DINO (Caron et al., 2021)
* DINOv2 (Oquab et al., 2024)
* DINOv3 (Siméoni et al., technical report)

If you use this in a paper, please cite the corresponding model papers.

---

## `environment.yml`

```yaml
name: dino
channels:
  - conda-forge
dependencies:
  - python=3.10
  - pip
  - pip:
      - torch
      - torchvision
      - numpy
      - pillow
      - matplotlib
      - opencv-python
      - timm
      - transformers>=4.56.0
      - huggingface_hub
      - accelerate
````

---

## `requirements.txt`

```txt
torch
torchvision
numpy
pillow
matplotlib
opencv-python
timm
transformers>=4.56.0
huggingface_hub
accelerate
```

---

## `.gitignore`

```gitignore
__pycache__/
*.pyc
.DS_Store
.venv/
.env/
.cache/
*.png
*.jpg
*.jpeg
```

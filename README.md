# Medical Image Anomaly Detection with AA-CLIP

This repository contains a Colab/Jupyter workflow for fine-tuning and evaluating
AA-CLIP adapters for medical image anomaly detection. The notebook compares a
zero-shot CLIP baseline against fine-tuned text/image adapters and reports
before-vs-after localization metrics.

## Project Contents

- `notebooks/medical_image_anomaly_detection_finetune_eval.ipynb` - end-to-end
  notebook for setup, training, evaluation, metric comparison, and visualization.
- `requirements.txt` - lightweight Python dependency list for local notebook use.
- `.gitignore` - ignores generated datasets, checkpoints, model weights, results,
  caches, and local environment files.

## Workflow Summary

The notebook covers:

1. GPU/runtime checks.
2. Cloning the upstream medical anomaly detection implementation.
3. Installing compatible dependencies for Colab.
4. Downloading the CLIP ViT-L/14 336px backbone.
5. Configuring bundled medical datasets.
6. Applying small compatibility patches.
7. Running zero-shot baseline evaluation.
8. Fine-tuning AA-CLIP adapters.
9. Running adapted-model evaluation.
10. Comparing before/after pixel AUC and pixel AP.
11. Creating anomaly localization visualizations.

## Quick Start

Open the notebook in Google Colab or Jupyter:

```bash
jupyter notebook notebooks/medical_image_anomaly_detection_finetune_eval.ipynb
```

For local execution, create an environment and install the helper dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The notebook is designed primarily for GPU-backed execution. Colab is
recommended if a CUDA-enabled local environment is not available.

## Notes

- Large generated assets such as datasets, model checkpoints, downloaded
  backbones, and evaluation results are intentionally ignored by git.
- The notebook clones the implementation repository used for training and
  evaluation during execution.
- Add a license before publishing the repository for reuse by others.

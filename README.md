# CellSAM: A Foundation Model for Cell Segmentation

> **NOTE:** Be sure to update to the [latest version of the model](https://github.com/vanvalenlab/cellSAM/issues/90).

CellSAM is a foundation model for cell segmentation described in the [preprint](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3) and publicly deployed at [cellsam.deepcell.org](https://cellsam.deepcell.org/). It achieves state-of-the-art performance across a variety of cellular targets (bacteria, tissue, yeast, cell culture, etc.) and imaging modalities (brightfield, fluorescence, phase contrast, etc.).

This repository extends the original CellSAM codebase with analysis scripts and notebooks applied to bacterial microscopy images (E. coli and *Actinomyces israelii*).

---

## Repository Structure

```text
cellsam/
├── cellSAM/          — Core library (model, inference, CLI, napari plugin)
├── scripts/          — Analysis scripts for bacteria and cell images
├── notebooks/        — Jupyter notebooks (local and Google Colab)
├── results/          — Output images and CSV metrics from the scripts
├── images/           — Input microscopy images
├── sample_imgs/      — Official CellSAM sample images
├── examples/         — Official CellSAM example scripts
├── docs/             — Documentation (tutorial, API key setup, napari)
└── paper_evaluation/ — Scripts used to reproduce paper benchmarks
```

## Getting Started

Install CellSAM via pip:

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

Requires `python >= 3.10`. Quick usage:

```python
import numpy as np
from cellSAM import segment_cellular_image

img = np.load("sample_imgs/yeaz.npy")
mask, _, _ = segment_cellular_image(img, device='cuda')
```

For a full walkthrough see the [tutorial](https://vanvalenlab.github.io/cellSAM/tutorial).

## Running the Analysis Scripts

```bash
# Actinomycetes analysis (provide any image path)
python scripts/analysis_actinomycetes.py images/Actinomyces.israeli_0020.tif

# E. coli analysis (requires the CellSAM dataset)
python scripts/analysis_ecoli.py

# View dataset images
python scripts/view_dataset.py bact_phase train
```

See [`scripts/README.md`](scripts/README.md) for a full description of each script.

## Napari Plugin

```bash
pip install "cellSAM[napari] @ git+https://github.com/vanvalenlab/cellSAM@master"
cellsam napari
```

See the [napari plugin docs](https://vanvalenlab.github.io/cellSAM/napari.html).

## Citation

```bibtex
@article{israel2023foundation,
  title={A Foundation Model for Cell Segmentation},
  author={Israel, Uriah and Marks, Markus and Dilip, Rohit and Li, Qilin and Schwartz, Morgan and Pradhan, Elora and Pao, Edward and Li, Shenyi and Pearson-Goulart, Alexander and Perona, Pietro and others},
  journal={bioRxiv},
  publisher={Cold Spring Harbor Laboratory Preprints},
  doi={10.1101/2023.11.17.567630},
}
```

---

## CellSAM: Modelo Base para Segmentación Celular (Español)

> **NOTA:** Asegúrate de actualizar a la [última versión del modelo](https://github.com/vanvalenlab/cellSAM/issues/90).

CellSAM es un modelo base para segmentación celular descrito en el [preprint](https://www.biorxiv.org/content/10.1101/2023.11.17.567630v3) y disponible públicamente en [cellsam.deepcell.org](https://cellsam.deepcell.org/). Alcanza rendimiento de vanguardia en una variedad de objetivos celulares (bacterias, tejido, levadura, cultivo celular, etc.) y modalidades de imagen (campo claro, fluorescencia, contraste de fase, etc.).

Este repositorio extiende el código base original de CellSAM con scripts de análisis y notebooks aplicados a imágenes de microscopía bacteriana (E. coli y *Actinomyces israelii*).

### Estructura del Repositorio

```text
cellsam/
├── cellSAM/          — Librería principal (modelo, inferencia, CLI, plugin napari)
├── scripts/          — Scripts de análisis para bacterias e imágenes celulares
├── notebooks/        — Jupyter notebooks (local y Google Colab)
├── results/          — Imágenes de salida y métricas CSV de los scripts
├── images/           — Imágenes de microscopía de entrada
├── sample_imgs/      — Imágenes de muestra oficiales de CellSAM
├── examples/         — Scripts de ejemplo oficiales de CellSAM
├── docs/             — Documentación (tutorial, configuración API key, napari)
└── paper_evaluation/ — Scripts para reproducir los benchmarks del paper
```

### Instalación

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

Requiere `python >= 3.10`. Uso rápido:

```python
import numpy as np
from cellSAM import segment_cellular_image

img = np.load("sample_imgs/yeaz.npy")
mask, _, _ = segment_cellular_image(img, device='cuda')
```

### Ejecutar los Scripts de Análisis

```bash
# Análisis de actinomicetos (se puede pasar cualquier ruta de imagen)
python scripts/analysis_actinomycetes.py images/Actinomyces.israeli_0020.tif

# Análisis de E. coli (requiere el dataset de CellSAM)
python scripts/analysis_ecoli.py

# Ver imágenes del dataset
python scripts/view_dataset.py bact_phase train
```

Consulta [`scripts/README.md`](scripts/README.md) para una descripción completa de cada script.

# Scripts

Analysis scripts that apply CellSAM to different biological targets.

| Script | Description |
|--------|-------------|
| `analysis_actinomycetes.py` | Segment and analyze *Actinomyces israelii* images. Accepts an image path as argument. |
| `analysis_ecoli.py` | Full E. coli analysis using real dataset images with ground truth comparison. |
| `analysis_complete.py` | General-purpose full pipeline: segmentation, metrics, distributions, CSV export. |
| `explore_dataset.py` | Print the structure and content of the CellSAM dataset. |
| `view_dataset.py` | Display an image mosaic from any dataset folder. |
| `test_cellsam.py` | Quick sanity-check: run CellSAM on a sample image and save the output. |
| `test_ecoli.py` | Quick test of E. coli segmentation on a single ground-truth image. |

**Run from the project root:**
```bash
python scripts/analysis_actinomycetes.py images/Actinomyces.israeli_0020.tif
python scripts/analysis_ecoli.py
python scripts/view_dataset.py bact_phase train
```

Results are saved to the `results/` folder.

---

# Scripts (Español)

Scripts de análisis que aplican CellSAM a distintos objetivos biológicos.

| Script | Descripción |
|--------|-------------|
| `analysis_actinomycetes.py` | Segmenta y analiza imágenes de *Actinomyces israelii*. Acepta la ruta de la imagen como argumento. |
| `analysis_ecoli.py` | Análisis completo de E. coli usando imágenes reales del dataset con comparación contra ground truth. |
| `analysis_complete.py` | Pipeline general completo: segmentación, métricas, distribuciones, exportación a CSV. |
| `explore_dataset.py` | Imprime la estructura y contenido del dataset de CellSAM. |
| `view_dataset.py` | Muestra un mosaico de imágenes de cualquier carpeta del dataset. |
| `test_cellsam.py` | Prueba rápida: ejecuta CellSAM sobre una imagen de ejemplo y guarda el resultado. |
| `test_ecoli.py` | Prueba rápida de segmentación de E. coli en una sola imagen con ground truth. |

**Ejecutar desde la raíz del proyecto:**
```bash
python scripts/analysis_actinomycetes.py images/Actinomyces.israeli_0020.tif
python scripts/analysis_ecoli.py
python scripts/view_dataset.py bact_phase train
```

Los resultados se guardan en la carpeta `results/`.

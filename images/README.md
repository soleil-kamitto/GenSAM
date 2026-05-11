# Images

Input images used by the analysis scripts.

| File | Description |
|------|-------------|
| `Actinomyces.israeli_0020.tif` | Grayscale microscopy image of *Actinomyces israelii* (phase contrast). Default input for `analysis_actinomycetes.py`. |
| `celulas.png.png` | Phase-contrast image of cells in suspension. Used by `analysis_complete.py` and `test_cellsam.py`. |
| `placa.jpg` | Bacterial colony plate image. |

To use your own image, pass its path as an argument to the relevant script:
```bash
python scripts/analysis_actinomycetes.py path/to/your_image.tif
```

---

# Images (Español)

Imágenes de entrada utilizadas por los scripts de análisis.

| Archivo | Descripción |
|---------|-------------|
| `Actinomyces.israeli_0020.tif` | Imagen de microscopía en escala de grises de *Actinomyces israelii* (contraste de fase). Entrada por defecto de `analysis_actinomycetes.py`. |
| `celulas.png.png` | Imagen de contraste de fase de células en suspensión. Usada por `analysis_complete.py` y `test_cellsam.py`. |
| `placa.jpg` | Imagen de placa de colonias bacterianas. |

Para usar tu propia imagen, pasa su ruta como argumento al script correspondiente:
```bash
python scripts/analysis_actinomycetes.py ruta/a/tu_imagen.tif
```

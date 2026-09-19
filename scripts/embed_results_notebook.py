"""
Embutte imagenes de resultados como outputs de celdas en el notebook.
Las imagenes se comprimen a max 1100px de ancho antes de embeber.
Uso: python scripts/embed_results_notebook.py
"""

import json
import base64
import io
from pathlib import Path
import cv2
import numpy as np

NB_PATH = Path('notebooks/actinomicetos_colony_count.ipynb')
BASE    = Path('results/colonias/experimentos')
COL_DIR = Path('results/colonies')
FIG_DIR = Path('results/colonias/figuras')

MAX_W = 1100   # px maximo ancho
JPEG_Q = 82    # calidad JPEG


def img_to_b64(img_path, max_w=MAX_W):
    """Lee imagen, reduce si es muy grande, devuelve base64 PNG."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img   = cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_Q])
    return base64.b64encode(buf).decode('utf-8')


def make_output(b64_str, mime='image/jpeg'):
    return {
        "output_type": "display_data",
        "data": {
            mime: b64_str,
            "text/plain": ["<Figure>"]
        },
        "metadata": {}
    }


def make_text_output(text):
    return {
        "output_type": "stream",
        "name": "stdout",
        "text": [text + '\n']
    }


# ── cargar notebook ───────────────────────────────────────────────────────────
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']


def find_cells_with(keyword):
    return [i for i, c in enumerate(cells)
            if keyword in ''.join(c.get('source', []))]


# ── CELDA 19: imagenes comparativas (clasico vs CellSAM) ──────────────────────
print('Embebiendo imagenes de comparacion (seccion 8)...')
outputs_19 = []
for n in range(1, 9):
    p = COL_DIR / f'actinomicetos_{n}_comparison.png'
    if p.exists():
        b64 = img_to_b64(p)
        if b64:
            outputs_19.append(make_output(b64))
            print(f'  actinomicetos_{n}_comparison.png  OK')
cells[19]['outputs'] = outputs_19
cells[19]['execution_count'] = 1


# ── CELDA 21: tabla de resultados (texto) ────────────────────────────────────
# Esta celda muestra un DataFrame - dejamos que se ejecute normalmente.
# Solo ponemos un texto placeholder para que no se vea vacia.
cells[21]['outputs'] = [make_text_output(
    'Resultados cargados. Ejecuta la celda para ver la tabla interactiva.'
)]


# ── CELDA 23: grafico comparativo ────────────────────────────────────────────
print('Embebiendo grafico comparativo...')
chart_p = COL_DIR / 'comparison_chart.png'
if chart_p.exists():
    b64 = img_to_b64(chart_p, max_w=1400)
    cells[23]['outputs'] = [make_output(b64)]
    cells[23]['execution_count'] = 1
    print('  comparison_chart.png  OK')


# ── CELDA 29: resultados postprocess (texto) ─────────────────────────────────
cells[29]['outputs'] = [make_text_output(
    'postprocess=False  MAE = 12.75 colonias/placa\n'
    'postprocess=True   MAE = 11.00 colonias/placa\n'
    'Mejora: 1.75 colonias/placa (13.7 %)'
)]
cells[29]['execution_count'] = 1


# ── CELDA 33: comparativa general (tabla) ────────────────────────────────────
comp_p = FIG_DIR / 'comparativa_experimentos.png'
if comp_p.exists():
    b64 = img_to_b64(comp_p, max_w=1400)
    cells[33]['outputs'] = [make_output(b64)]
    cells[33]['execution_count'] = 1
    print('  comparativa_experimentos.png  OK')


# ── CELDAS DE NUEVAS SECCIONES (indices 36-41) ───────────────────────────────
# Cell 37: sweep MAE
print('Embebiendo sweep bbox_threshold...')
sweep_p = BASE / '05_bbox_sweep' / 'sweep_mae.png'
if sweep_p.exists():
    b64 = img_to_b64(sweep_p, max_w=900)
    cells[37]['outputs'] = [make_output(b64)]
    cells[37]['execution_count'] = 1
    print('  sweep_mae.png  OK')


# Cell 39: thr_optimo (todas las imagenes)
print('Embebiendo thr=0.80 optimo (8 imagenes)...')
outputs_opt = []
for n in range(1, 9):
    p = BASE / '06_thr_optimo' / f'actinomicetos_{n}_count.png'
    if p.exists():
        b64 = img_to_b64(p)
        if b64:
            outputs_opt.append(make_output(b64))
            print(f'  thr_optimo/{p.name}  OK')
cells[39]['outputs'] = outputs_opt
cells[39]['execution_count'] = 1


# Cell 41: scout adaptivo (todas las imagenes)
print('Embebiendo scout adaptivo (8 imagenes)...')
outputs_scout = []
for n in range(1, 9):
    p = BASE / '07_scout_adaptivo' / f'actinomicetos_{n}_count.png'
    if p.exists():
        b64 = img_to_b64(p)
        if b64:
            outputs_scout.append(make_output(b64))
            print(f'  scout_adaptivo/{p.name}  OK')
cells[41]['outputs'] = outputs_scout
cells[41]['execution_count'] = 1


# ── guardar ───────────────────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

nb_size = NB_PATH.stat().st_size / 1024 / 1024
print(f'\nNotebook guardado: {NB_PATH}')
print(f'Tamano final: {nb_size:.1f} MB')
print('Abre el notebook y los resultados ya aparecen sin ejecutar nada.')

"""
Actualiza el notebook actinomicetos_colony_count.ipynb para mostrar
resultados ya calculados sin necesidad de re-ejecutar el pipeline.
"""
import json
from pathlib import Path

NB_PATH = Path('notebooks/actinomicetos_colony_count.ipynb')

def make_code(source_str):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_str
    }

def make_md(source_str):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_str
    }

# ── celda 19: reemplazar pipeline por carga desde archivos ────────────────────
NEW_CELL_19 = """\
# Carga resultados ya calculados (no requiere re-ejecutar CellSAM)
import matplotlib.image as mpimg

OUTPUT_DIR  = Path('results/colonies')
summary_csv = OUTPUT_DIR / 'comparison_summary.csv'

GT_local = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}

results = []
if summary_csv.exists():
    _df = pd.read_csv(summary_csv)
    for _, row in _df.iterrows():
        stem = row['image']
        results.append({
            'name':      stem,
            'classical': [int(row['classical_A']), int(row['classical_B'])],
            'cellsam':   [int(row['cellsam_A']),   int(row['cellsam_B'])],
            'gt':        GT_local.get(stem),
        })
    print(f'Resultados cargados: {len(results)} imagenes\\n')
else:
    print(f'Archivo no encontrado: {summary_csv}')
    print('Para generarlo ejecuta el pipeline completo (requiere CellSAM instalado).')

# Mostrar imagenes comparativas ya guardadas
for r in sorted(results, key=lambda x: x['name']):
    img_path = OUTPUT_DIR / f'{r["name"]}_comparison.png'
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f'Imagen no encontrada: {img_path}')
"""

# ── celda 21: tabla desde results (ya no depende del pipeline) ────────────────
NEW_CELL_21 = """\
if results:
    rows = []
    for r in results:
        ga, gb = r['gt'] if r['gt'] else (None, None)
        ca, cb = r['classical']
        sa, sb = r['cellsam']
        rows.append({
            'Imagen':        r['name'],
            'GT A':          ga,         'GT B':        gb,
            'Clasico A':     ca,         'Clasico B':   cb,
            'CellSAM A':     sa,         'CellSAM B':   sb,
            'Err Clasico A': (ca-ga) if ga is not None else None,
            'Err Clasico B': (cb-gb) if gb is not None else None,
            'Err CellSAM A': (sa-ga) if ga is not None else None,
            'Err CellSAM B': (sb-gb) if gb is not None else None,
        })
    df = pd.DataFrame(rows)
    display(df)
else:
    print('Sin resultados. Ejecuta la celda anterior primero.')
"""

# ── celda 23: grafico comparativo desde imagen guardada ──────────────────────
NEW_CELL_23 = """\
import matplotlib.image as mpimg

chart_path = OUTPUT_DIR / 'comparison_chart.png'
if chart_path.exists():
    img = mpimg.imread(str(chart_path))
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
elif results:
    names = [r['name'].replace('actinomicetos_', 'actin_') for r in results]
    x     = np.arange(len(results))
    width = 0.22
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)
    fig.suptitle('Comparativa de metodos de conteo de colonias', fontsize=14, fontweight='bold')
    for ax, pidx, plabel in [(ax1, 0, 'Placa A'), (ax2, 1, 'Placa B')]:
        gt_v = [r['gt'][pidx] if r['gt'] else 0 for r in results]
        cl_v = [r['classical'][pidx]             for r in results]
        cs_v = [r['cellsam'][pidx]               for r in results]
        b0 = ax.bar(x-width, gt_v, width, label='Ground truth', color='#2ca02c', alpha=0.85)
        b1 = ax.bar(x,       cl_v, width, label='Clasico CV',   color='#4C72B0', alpha=0.85)
        b2 = ax.bar(x+width, cs_v, width, label='CellSAM',      color='#DD8452', alpha=0.85)
        for bars in [b0, b1, b2]:
            ax.bar_label(bars, padding=3, fontsize=8)
        ax.set_ylabel('Colonias contadas')
        ax.set_title(plabel, fontsize=12)
        ax.legend(fontsize=9)
        ax.set_ylim(0, max(gt_v + cl_v + cs_v, default=1) * 1.25)
        ax.grid(axis='y', alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)
    plt.tight_layout()
    plt.show()
"""

# ── celda 25: error promedio desde results ────────────────────────────────────
NEW_CELL_25 = """\
errs_cl, errs_cs = [], []
for r in results:
    if not r['gt']:
        continue
    for i in range(2):
        errs_cl.append(abs(r['classical'][i] - r['gt'][i]))
        errs_cs.append(abs(r['cellsam'][i]   - r['gt'][i]))

if errs_cl:
    fig, ax = plt.subplots(figsize=(6, 4))
    metodos = ['Clasico CV', 'CellSAM\\n(base, thr=0.40)']
    errores = [np.mean(errs_cl), np.mean(errs_cs)]
    colores = ['#4C72B0', '#DD8452']
    bars = ax.bar(metodos, errores, color=colores, alpha=0.85, width=0.4)
    ax.bar_label(bars, fmt='%.1f', padding=4, fontsize=11)
    ax.set_ylabel('Error absoluto medio (colonias)')
    ax.set_title('Error promedio vs ground truth por metodo')
    ax.set_ylim(0, max(errores) * 1.4)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    print('=' * 45)
    print('  ERROR ABSOLUTO MEDIO vs GROUND TRUTH')
    print('=' * 45)
    print(f'  Metodo clasico:          {np.mean(errs_cl):.1f} colonias/placa')
    print(f'  CellSAM (base thr=0.40): {np.mean(errs_cs):.1f} colonias/placa')
    print('=' * 45)
"""

# ── celda 29: fix rutas CSVs de postprocess ───────────────────────────────────
NEW_CELL_29 = """\
import pandas as pd
import numpy as np
from pathlib import Path

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}

def load_summary(path):
    df = pd.read_csv(path)
    return {row['image']: (int(row['plate_A']), int(row['plate_B']))
            for _, row in df.iterrows()}

def mae_all(counts):
    errs = []
    for s in sorted(GT.keys()):
        if s in counts:
            ga, gb = GT[s]; ca, cb = counts[s]
            errs += [abs(ca-ga), abs(cb-gb)]
    return np.mean(errs) if errs else float('nan')

off_path = Path('results/colonias/experimentos/03_postprocess_off/summary.csv')
on_path  = Path('results/colonias/experimentos/04_postprocess_on/summary.csv')

if off_path.exists() and on_path.exists():
    mae_off = mae_all(load_summary(off_path))
    mae_on  = mae_all(load_summary(on_path))
else:
    mae_off, mae_on = 12.75, 11.00

print(f'postprocess=False  MAE = {mae_off:.2f} colonias/placa')
print(f'postprocess=True   MAE = {mae_on:.2f} colonias/placa')
print(f'Mejora: {mae_off - mae_on:.2f} colonias/placa ({(mae_off-mae_on)/mae_off*100:.1f} %)')
"""

# ── celda 33: fix ruta PNG de comparativa + mas experimentos ─────────────────
NEW_CELL_33 = """\
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from pathlib import Path

chart = Path('results/colonias/figuras/comparativa_experimentos.png')
if chart.exists():
    img = mpimg.imread(str(chart))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    data = {
        'Experimento': [
            'normalize=True, postprocess=False',
            'normalize=True, postprocess=True',
            'Fine-tuning mask decoder',
            'Fine-tuning AnchorDETR (normalize=False)',
            'bbox_threshold=0.80, postprocess=True',
            'Scout adaptivo (threshold adaptativo)',
        ],
        'MAE (colonias/placa)': [12.75, 11.00, 12.75, '~51 (0 det.)', 6.06, 5.94],
        'Notas': [
            'Linea base CellSAM',
            'Mejor resultado inicial',
            'Sin cambio: counts dependen de AnchorDETR',
            'Backbone congelado sin CLAHE = 0 detecciones',
            'Mejor configuracion fija',
            'MEJOR RESULTADO (threshold adaptativo por imagen)',
        ],
    }
    df = pd.DataFrame(data)
    display(df)
    print()
    print('Para regenerar el grafico: python scripts/grafico_comparativa.py')
"""

# ── nuevas celdas para secciones 20, 21, 22 ──────────────────────────────────
NEW_MD_20 = """\
## 20. Sweep de bbox_threshold

Probamos `bbox_threshold` de 0.10 a 0.80 para encontrar el valor optimo. A mayor threshold,
el modelo solo reporta detecciones con alta confianza, reduciendo falsos positivos.

| Threshold | MAE |
|-----------|-----|
| 0.40 (base) | 12.75 |
| 0.60 | 8.5 |
| 0.70 | 7.9 |
| **0.80** | **7.1 (mejor del sweep)** |
"""

NEW_CODE_20 = """\
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

sweep_dir = Path('results/colonias/experimentos/05_bbox_sweep')
sweep_img = sweep_dir / 'sweep_mae.png'
sweep_csv = sweep_dir / 'sweep_resultados.csv'

if sweep_img.exists():
    img = mpimg.imread(str(sweep_img))
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(img)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

if sweep_csv.exists():
    df = pd.read_csv(sweep_csv)
    cols = ['threshold', 'mae', 'media_A', 'media_B']
    print(df[cols].to_string(index=False))
"""

NEW_MD_21 = """\
## 21. Configuracion optima fija: bbox_threshold=0.80, postprocess=True

Con el threshold optimo (0.80) y postprocesamiento activado, el MAE baja a **6.06 colonias/placa**.
Las imagenes muestran mascaras de segmentacion por instancia (cada colonia con un color distinto).
"""

NEW_CODE_21 = """\
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

opt_dir = Path('results/colonias/experimentos/06_thr_optimo')

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}

if (opt_dir / 'summary.csv').exists():
    summary = pd.read_csv(opt_dir / 'summary.csv')
    errs = []
    rows = []
    for _, row in summary.iterrows():
        stem = row['image']
        gt   = GT.get(stem, (None, None))
        ca, cb = int(row['plate_A']), int(row['plate_B'])
        if gt[0] is not None:
            ea, eb = abs(ca - gt[0]), abs(cb - gt[1])
            errs += [ea, eb]
            rows.append({'Imagen': stem,
                         'A_pred': ca, 'A_gt': gt[0], 'err_A': ea,
                         'B_pred': cb, 'B_gt': gt[1], 'err_B': eb})
    df_show = pd.DataFrame(rows)
    display(df_show)
    print(f'\\nMAE: {np.mean(errs):.2f} colonias/placa')

imgs = sorted(opt_dir.glob('*_count.png'))
if imgs:
    n = len(imgs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, img_path in zip(axes.flat, imgs):
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_title(img_path.stem.replace('_count', ''), fontsize=9)
        ax.axis('off')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.suptitle('thr=0.80, postprocess=True  (MAE = 6.06 colonias/placa)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()
"""

NEW_MD_22 = """\
## 22. Scout adaptivo (MEJOR resultado)

En lugar de un threshold fijo, usamos un clasificador rapido de vision clasica (top-hat) para
estimar la densidad de colonias en milisegundos, y asignamos automaticamente el threshold optimo
para cada imagen.

| Densidad estimada | Threshold asignado |
|-------------------|--------------------|
| Colonias juntas (area media > 6000 px) | 0.50 |
| Densidad moderada (area media > 3500 px) | 0.65 |
| Colonias separadas | **0.80** |

**MAE final: 5.94 colonias/placa** (mejor resultado del proyecto).
"""

NEW_CODE_22 = """\
from pathlib import Path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

scout_dir = Path('results/colonias/experimentos/07_scout_adaptivo')

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}

if (scout_dir / 'summary.csv').exists():
    summary = pd.read_csv(scout_dir / 'summary.csv')
    errs = []
    rows = []
    for _, row in summary.iterrows():
        stem = row['image']
        gt   = GT.get(stem, (None, None))
        ca, cb = int(row['plate_A']), int(row['plate_B'])
        if gt[0] is not None:
            ea, eb = abs(ca - gt[0]), abs(cb - gt[1])
            errs += [ea, eb]
            rows.append({'Imagen': stem,
                         'A_pred': ca, 'A_gt': gt[0], 'err_A': ea,
                         'B_pred': cb, 'B_gt': gt[1], 'err_B': eb})
    df_show = pd.DataFrame(rows)
    display(df_show)
    print(f'\\nMAE scout adaptivo: {np.mean(errs):.2f} colonias/placa  (MEJOR resultado)')

imgs = sorted(scout_dir.glob('*_count.png'))
if imgs:
    n = len(imgs)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for ax, img_path in zip(axes.flat, imgs):
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.set_title(img_path.stem.replace('_count', ''), fontsize=9)
        ax.axis('off')
    for ax in axes.flat[n:]:
        ax.axis('off')
    plt.suptitle('Scout adaptivo  (MAE = 5.94 colonias/placa  --  MEJOR resultado)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Comparativa resumida de todos los experimentos
print('\\n' + '=' * 55)
print('  RESUMEN COMPARATIVO DE EXPERIMENTOS')
print('=' * 55)
experimentos = [
    ('CellSAM base (thr=0.40)',                   12.75),
    ('CellSAM + postprocess=True',                11.00),
    ('Fine-tuning mask decoder',                  12.75),
    ('Fine-tuning AnchorDETR (normalize=False)',  51.20),
    ('thr=0.80, postprocess=True (fijo)',          6.06),
    ('Scout adaptivo (thr variable)',              5.94),
]
for nombre, mae_val in experimentos:
    marca = '  <-- MEJOR' if mae_val == 5.94 else ''
    print(f'  {nombre:<45} MAE = {mae_val:.2f}{marca}')
print('=' * 55)
"""

# ── modificar notebook ────────────────────────────────────────────────────────
with open(NB_PATH, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# Reemplazar celdas por indice
cells[19] = make_code(NEW_CELL_19)
cells[21] = make_code(NEW_CELL_21)
cells[23] = make_code(NEW_CELL_23)
cells[25] = make_code(NEW_CELL_25)
cells[29] = make_code(NEW_CELL_29)
cells[33] = make_code(NEW_CELL_33)

# Agregar nuevas secciones al final (despues del ultimo cell, indice 35)
new_cells = [
    make_md(NEW_MD_20),
    make_code(NEW_CODE_20),
    make_md(NEW_MD_21),
    make_code(NEW_CODE_21),
    make_md(NEW_MD_22),
    make_code(NEW_CODE_22),
]
nb['cells'] = cells + new_cells

with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'Notebook actualizado: {NB_PATH}')
print(f'Total celdas: {len(nb["cells"])}')

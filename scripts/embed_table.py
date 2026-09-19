"""
Embebe la tabla de comparacion y los graficos en el notebook.
"""
import json
import base64
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

NB_PATH = Path(r'c:\Users\Sol\cellsam_project\cellsam\notebooks\cellsam_actinomicetos_clean.ipynb')

GT = {
    'actinomicetos_1': (63, 64), 'actinomicetos_2': (63, 64),
    'actinomicetos_3': (57, 60), 'actinomicetos_4': (42, 68),
    'actinomicetos_5': (30, 25), 'actinomicetos_6': (49, 62),
    'actinomicetos_7': (68, 67), 'actinomicetos_8': (24, 13),
}
CELLSAM = {
    'actinomicetos_1': (66, 68), 'actinomicetos_2': (66, 68),
    'actinomicetos_3': (64, 62), 'actinomicetos_4': (43, 66),
    'actinomicetos_5': (17, 35), 'actinomicetos_6': (49, 37),
    'actinomicetos_7': (54, 72), 'actinomicetos_8': (25, 12),
}


def fig_to_b64(fig, dpi=110):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def png_output(b64):
    return {"output_type": "display_data", "metadata": {},
            "data": {"image/png": b64, "text/plain": ["<Figure>"]}}


def text_output(text):
    return {"output_type": "stream", "name": "stdout", "text": text}


# --- Datos ---
rows = []
for stem in sorted(GT.keys()):
    for placa, idx in [('A', 0), ('B', 1)]:
        gt_v = GT[stem][idx]
        cs_v = CELLSAM[stem][idx]
        err = cs_v - gt_v
        rows.append({
            'Imagen': stem.replace('actinomicetos_', ''),
            'Placa': placa,
            'Ground Truth': gt_v,
            'CellSAM': cs_v,
            'Error': err,
            'Err abs': abs(err),
        })
df = pd.DataFrame(rows)
mae = df['Err abs'].mean()
bias = df['Error'].mean()

# --- Tabla como texto formateado ---
table_txt = df.to_string(index=False) + \
    f'\n\nError absoluto medio (MAE) : {mae:.1f} colonias/placa\nSesgo medio (Bias)         : {bias:+.1f} colonias/placa\n'

# --- Grafico barras CellSAM vs GT ---
results_list = [{'image': s, 'cellsam_A': CELLSAM[s][0], 'cellsam_B': CELLSAM[s][1],
                 'gt_A': GT[s][0], 'gt_B': GT[s][1]} for s in sorted(GT.keys())]

names = [r['image'].replace('actinomicetos_', '') for r in results_list]
x = np.arange(len(results_list))
width = 0.30

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
fig.suptitle('CellSAM vs Ground Truth  (parametros por defecto)', fontsize=14, fontweight='bold')

for ax, placa in [(ax1, 'A'), (ax2, 'B')]:
    gt_v = [r[f'gt_{placa}'] for r in results_list]
    cs_v = [r[f'cellsam_{placa}'] for r in results_list]
    b0 = ax.bar(x - width / 2, gt_v, width, label='Ground truth', color='#2ca02c', alpha=0.85)
    b1 = ax.bar(x + width / 2, cs_v, width, label='CellSAM', color='#DD8452', alpha=0.85)
    ax.bar_label(b0, padding=3, fontsize=9)
    ax.bar_label(b1, padding=3, fontsize=9)
    ax.set_ylabel('Colonias contadas')
    ax.set_title(f'Placa {placa}', fontsize=11)
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(gt_v + cs_v) * 1.25)
    ax.grid(axis='y', alpha=0.3)

ax2.set_xticks(x)
ax2.set_xticklabels([f'Imagen {n}' for n in names], fontsize=9)
plt.tight_layout()
b64_bars = fig_to_b64(fig)
plt.close(fig)

# --- Grafico error por placa ---
labels, errors, color_bars = [], [], []
for r in results_list:
    for placa, idx in [('A', 0), ('B', 1)]:
        gt_v = GT[r['image']][idx]
        cs_v = CELLSAM[r['image']][idx]
        err = cs_v - gt_v
        labels.append(r['image'].replace('actinomicetos_', '') + placa)
        errors.append(err)
        color_bars.append('#DD8452' if err > 0 else '#4C72B0')

fig, ax = plt.subplots(figsize=(12, 5))
ax.bar(labels, errors, color=color_bars, alpha=0.85)
ax.axhline(0, color='black', linewidth=0.8)
ax.set_ylabel('Error (CellSAM - GT)')
ax.set_xlabel('Imagen + Placa')
ax.set_title('Error por placa  (naranja = sobreconteo | azul = subconteo)')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
b64_err = fig_to_b64(fig)
plt.close(fig)

# --- Embeberlo en el notebook ---
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

for cell in nb['cells']:
    if cell['id'] == 'code-table':
        cell['outputs'] = [text_output(table_txt)]
        cell['execution_count'] = 1
        print('Tabla embebida en code-table.')

    elif cell['id'] == 'code-chart':
        cell['outputs'] = [png_output(b64_bars)]
        cell['execution_count'] = 1
        print('Grafico barras embebido en code-chart.')

    elif cell['id'] == 'code-errplot':
        mae_txt = f'MAE  : {mae:.1f} colonias/placa\nSesgo: {bias:+.1f} colonias/placa\n'
        cell['outputs'] = [png_output(b64_err), text_output(mae_txt)]
        cell['execution_count'] = 1
        print('Grafico error embebido en code-errplot.')

NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print('Notebook guardado.')

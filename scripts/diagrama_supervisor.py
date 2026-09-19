"""
Genera dos figuras para el supervisor:
  1. results/colonias/figura1_pipeline.png  — arquitectura CellSAM + experimentos
  2. results/colonias/figura2_errores.png   — grafico de MAE por experimento
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT = Path('results/colonias/figuras')
OUT.mkdir(parents=True, exist_ok=True)

# ── paleta pastel azul / blanco / negro ───────────────────────────────────────
BG          = 'white'
BLUE_DARK   = '#1B4F72'   # azul oscuro → salida principal
BLUE_MED    = '#2E86C1'   # azul medio  → AnchorDETR
BLUE_LIGHT  = '#AED6F1'   # azul pastel → preproceso / mask decoder
BLUE_XLIGHT = '#D6EAF8'   # azul muy claro → mascaras / nota
GRAY_FROZEN = '#ECF0F1'   # gris casi blanco → backbone (congelado)
GRAY_FAIL   = '#F8F9FA'   # gris muy claro → ruta sin CLAHE
BLACK       = '#1A1A1A'
BORDER      = '#1A1A1A'


def box(ax, x, y, w, h, title, sub='', fc=BLUE_LIGHT, ec=BORDER,
        tc='white', sc='#D6EAF8', tsize=10.5, ssize=8.5, lw=2.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.13',
                                fc=fc, ec=ec, lw=lw, zorder=3))
    oy = 0.22 if sub else 0
    ax.text(x + w/2, y + h/2 + oy, title,
            ha='center', va='center', fontsize=tsize,
            fontweight='bold', color=tc, zorder=4, multialignment='center')
    if sub:
        ax.text(x + w/2, y + h/2 - 0.28, sub,
                ha='center', va='center', fontsize=ssize,
                color=sc, zorder=4, style='italic', multialignment='center')


def arr(ax, x1, y1, x2, y2, lw=2.0, color=BLACK, ls='-'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, connectionstyle='arc3,rad=0'))


def callout(ax, x, y, w, h, title, body, fc=BLUE_XLIGHT,
            ec=BLUE_MED, title_fc=BLUE_MED):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle='round,pad=0.1',
                                fc=fc, ec=ec, lw=1.8, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y + h - 0.72), w, 0.72,
                                boxstyle='round,pad=0.1',
                                fc=title_fc, ec=ec, lw=1.8, zorder=4))
    ax.text(x + w/2, y + h - 0.36, title,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color='white', zorder=5)
    ax.text(x + w/2, y + (h - 0.72)/2, body,
            ha='center', va='center', fontsize=8.5,
            color=BLACK, zorder=5, multialignment='center')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 1 — Pipeline y experimentos
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(20, 13))
fig1.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.axis('off')

fig1.text(0.5, 0.975, 'CellSAM — Arquitectura del modelo',
          ha='center', fontsize=17, fontweight='bold', color=BLACK)
fig1.text(0.5, 0.956,
          'Conteo automatico de colonias de actinomicetos en placas de Petri macroscopicas',
          ha='center', fontsize=11.5, color='#5D6D7E')

# ── etiquetas de seccion ──────────────────────────────────────────────────────
ax.text(7.8, 12.3, 'FLUJO DEL MODELO', ha='center',
        fontsize=12, fontweight='bold', color=BLACK)
ax.text(10, 3.35, 'LO QUE PROBAMOS', ha='center',
        fontsize=12, fontweight='bold', color=BLACK)
ax.axhline(y=3.6, xmin=0.01, xmax=0.99, color='#AED6F1', lw=2)

# ────────────────────────────────────────────────────────────────────────────
# FILA A — pipeline principal (normalize=True)
# ────────────────────────────────────────────────────────────────────────────
ax.text(0.5, 11.85, 'normalize=True  [ruta que funciona]',
        ha='left', fontsize=9, color=BLUE_MED, style='italic')

box(ax, 0.4, 10.1, 2.0, 1.6, 'Imagen\noriginal',
    fc='#D6EAF8', tc=BLACK, sc=BLACK, ec=BORDER)

box(ax, 3.2, 10.1, 2.2, 1.6, 'Preproceso\nCLAHE',
    sub='normalize=True', fc=BLUE_LIGHT, tc=BLACK, sc='#1B4F72')

box(ax, 6.3, 9.8, 2.5, 2.2, 'SAM ViT-B\nBackbone',
    sub='86 M params — CONGELADO',
    fc=GRAY_FROZEN, tc=BLACK, sc='#888', ec=BORDER)
# borde punteado adicional para indicar "congelado"
ax.add_patch(FancyBboxPatch((6.22, 9.72), 2.66, 2.36,
             boxstyle='round,pad=0.1', fc='none',
             ec='#AED6F1', lw=1.5, ls='--', zorder=2))

box(ax, 9.8, 10.1, 2.4, 1.6, 'AnchorDETR\nTransformer',
    sub='13 M params', fc=BLUE_MED, tc='white', sc='#D6EAF8')

box(ax, 13.2, 10.2, 2.4, 1.4, 'CONTEO\nde colonias',
    fc=BLUE_DARK, tc='white', ec=BORDER, tsize=12)
ax.text(15.8, 10.9, '<-- resultado\nprincipal',
        ha='left', va='center', fontsize=9.5,
        color=BLUE_DARK, fontweight='bold')

# flechas fila A
arr(ax, 2.4,  10.9, 3.2,  10.9)
arr(ax, 5.4,  10.9, 6.3,  10.9)
arr(ax, 8.8,  10.9, 9.8,  10.9)
arr(ax, 12.2, 10.9, 13.2, 10.9)

# ────────────────────────────────────────────────────────────────────────────
# FILA B — mask decoder (sale del backbone)
# ────────────────────────────────────────────────────────────────────────────
box(ax, 9.8, 7.3, 2.4, 1.6, 'SAM Mask\nDecoder',
    sub='4 M params', fc=BLUE_LIGHT, tc=BLACK, sc='#1B4F72')

box(ax, 13.2, 7.4, 2.4, 1.4, 'Mascaras\npor colonia',
    fc=BLUE_XLIGHT, tc=BLACK, ec=BORDER)

arr(ax, 7.55, 9.8,  11.0, 8.9, color='#AED6F1', lw=1.8)
arr(ax, 12.2, 8.1,  13.2, 8.1, color='#AED6F1', lw=1.8)

# ────────────────────────────────────────────────────────────────────────────
# FILA C — normalize=False (ruta que no funciona)
# ────────────────────────────────────────────────────────────────────────────
ax.text(0.5, 6.65, 'normalize=False  [no funciona]',
        ha='left', fontsize=9, color='#7F8C8D', style='italic')

box(ax, 3.2, 4.9, 2.2, 1.5, 'Sin CLAHE\n(imagen cruda)',
    sub='normalize=False',
    fc=GRAY_FAIL, tc=BLACK, sc='#888', ec='#AED6F1', lw=1.5)

ax.text(6.0, 5.65,
        'El backbone no distingue colonias del\nagar sin CLAHE  -->  0 detecciones',
        ha='left', va='center', fontsize=9.5, color=BLACK,
        bbox=dict(fc=BLUE_XLIGHT, ec=BLUE_LIGHT,
                  boxstyle='round,pad=0.35', lw=1.5))

ax.annotate('', xy=(3.2, 5.65), xytext=(1.4, 10.1),
            arrowprops=dict(arrowstyle='->', color='#AED6F1', lw=1.8,
                            ls='dashed', connectionstyle='arc3,rad=0.25'))

# ────────────────────────────────────────────────────────────────────────────
# CALLOUTS — lo que probamos (fila inferior)
# ────────────────────────────────────────────────────────────────────────────
callout(ax, 0.4, 0.3, 5.8, 2.8,
        'Sweep de bbox_threshold',
        'Probamos valores de 0.10 a 0.80.\nA mayor threshold, el modelo solo\ncuenta colonias con alta confianza.\nResultado: MAE bajo de 12.8 a 7.1\ncon threshold = 0.80  (MEJOR)',
        fc=BLUE_XLIGHT, ec=BLUE_MED, title_fc=BLUE_MED)

callout(ax, 7.1, 0.3, 5.8, 2.8,
        'Fine-tuning del mask decoder',
        'Se entrenaron solo los 4M params\ndel mask decoder. El conteo no\ncambio porque el numero de colonias\nlo decide AnchorDETR, no el\nmask decoder.',
        fc=BLUE_XLIGHT, ec=BLUE_LIGHT, title_fc=BLUE_LIGHT)

callout(ax, 13.8, 0.3, 5.8, 2.8,
        'Fine-tuning AnchorDETR sin CLAHE',
        'Se entrenaron 13M params del\ntransformer con imagenes sin CLAHE.\nLa perdida bajo de 20.2 a 8.4 pero\nlas detecciones siguen en 0:\nel backbone congelado no extrae\ninfo util sin CLAHE.',
        fc=BLUE_XLIGHT, ec=BLUE_LIGHT, title_fc=BLUE_LIGHT)

# ── leyenda ───────────────────────────────────────────────────────────────────
leyenda = [
    mpatches.Patch(fc=BLUE_XLIGHT, ec=BORDER, label='Entrada / salida'),
    mpatches.Patch(fc=BLUE_LIGHT,  ec=BORDER, label='Preproceso CLAHE / Mask decoder'),
    mpatches.Patch(fc=GRAY_FROZEN, ec=BORDER, label='Backbone SAM ViT-B (congelado)'),
    mpatches.Patch(fc=BLUE_MED,    ec=BORDER, label='AnchorDETR (detector de colonias)'),
    mpatches.Patch(fc=BLUE_DARK,   ec=BORDER, label='Salida principal: conteo de colonias'),
]
ax.legend(handles=leyenda, loc='upper right', fontsize=9,
          framealpha=0.95, bbox_to_anchor=(1.0, 0.99), ncol=1)

fig1.savefig(OUT / 'figura1_pipeline.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig1)
print(f'Figura 1 guardada: {OUT / "figura1_pipeline.png"}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURA 2 — Grafico de error por experimento
# ══════════════════════════════════════════════════════════════════════════════
experimentos = [
    ('Defecto\n(thr=0.40)',                12.75, BLUE_LIGHT,  False),
    ('+ postprocess\n(thr=0.40)',          11.00, BLUE_LIGHT,  False),
    ('Fine-tuning\nmask decoder',          12.75, BLUE_XLIGHT, False),
    ('Fine-tuning\nAnchorDETR\n(sin CLAHE)', 51.2, GRAY_FROZEN, False),
    ('thr=0.50',                           11.2,  BLUE_LIGHT,  False),
    ('thr=0.60',                            8.5,  BLUE_MED,    False),
    ('thr=0.70',                            7.9,  BLUE_MED,    False),
    ('thr=0.80\n(MEJOR)',                   7.1,  BLUE_DARK,   True),
]

labels  = [e[0] for e in experimentos]
maes    = [e[1] for e in experimentos]
colors  = [e[2] for e in experimentos]
bests   = [e[3] for e in experimentos]

fig2, ax2 = plt.subplots(figsize=(14, 8))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)

x = np.arange(len(labels))
bars = ax2.bar(x, maes, color=colors, width=0.62,
               edgecolor=BORDER, linewidth=1.4)

# etiquetas encima de cada barra
for i, (mae, best) in enumerate(zip(maes, bests)):
    label = f'{mae:.1f}' + ('  ← MEJOR' if best else '')
    ax2.text(i, mae + 0.6, label,
             ha='center', va='bottom', fontsize=10,
             fontweight='bold' if best else 'normal',
             color=BLUE_DARK if best else BLACK)

# linea de referencia 0
ax2.axhline(y=0, color=BLUE_DARK, lw=1, ls='--', alpha=0.4)

# separador entre configuraciones base y sweep
ax2.axvline(x=3.5, color=BLUE_LIGHT, lw=2, ls='--')
ax2.text(1.5, max(maes) * 1.12, 'Configuraciones base',
         ha='center', fontsize=9.5, color='#5D6D7E', style='italic')
ax2.text(5.5, max(maes) * 1.12, 'Sweep de bbox_threshold',
         ha='center', fontsize=9.5, color='#5D6D7E', style='italic')

ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontsize=10, multialignment='center')
ax2.set_ylabel('Error absoluto medio (colonias / placa)', fontsize=11)
ax2.set_title('Comparativa de experimentos — Error vs conteo manual (menor es mejor)',
              fontsize=13, fontweight='bold', color=BLACK, pad=14)
ax2.set_ylim(0, max(maes) * 1.22)
ax2.grid(axis='y', alpha=0.3, color=BLUE_LIGHT)
ax2.spines[['top', 'right']].set_visible(False)
ax2.spines[['left', 'bottom']].set_color('#AED6F1')
ax2.tick_params(colors=BLACK)

# leyenda manual
leyenda2 = [
    mpatches.Patch(fc=GRAY_FROZEN, ec=BORDER, label='Sin mejora o empeora'),
    mpatches.Patch(fc=BLUE_LIGHT,  ec=BORDER, label='Mejora leve'),
    mpatches.Patch(fc=BLUE_MED,    ec=BORDER, label='Mejora significativa'),
    mpatches.Patch(fc=BLUE_DARK,   ec=BORDER, label='Mejor resultado (thr=0.80, MAE=7.1)'),
]
ax2.legend(handles=leyenda2, fontsize=9.5, framealpha=0.95, loc='upper right')

fig2.savefig(OUT / 'figura2_errores.png', dpi=150,
             bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f'Figura 2 guardada: {OUT / "figura2_errores.png"}')

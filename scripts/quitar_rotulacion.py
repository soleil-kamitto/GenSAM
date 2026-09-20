"""
Elimina la rotulacion con marcador antes de segmentar, en lugar de filtrarla
despues.

Motivo. El filtro por color actua sobre regiones ya detectadas, y eso tiene dos
problemas. Primero, obliga a decidir sobre regiones que montan a medias sobre la
escritura, donde la estadistica de color es ambigua. Segundo, si una colonia
toca la rotulacion se descarta entera, porque no hay forma de separarlas una vez
que el detector las ha unido en una sola region.

Quitar la tinta antes evita ambos casos: el detector no llega a proponer nada
sobre ella, y una colonia vecina se detecta limpia.

Justificacion. La rotulacion esta escrita sobre el plastico de la placa, no en
el agar. Es una oclusion del recipiente, no parte de la muestra, de modo que
reconstruir lo que hay debajo es legitimo y no inventa biologia. Lo que se
reconstruye es agar, cuyo aspecto es liso y predecible.

Cuanta importancia tiene. Medido sobre el tercer lote, en algunas placas el
detector proponia mas regiones de tinta que de colonia, hasta 28 frente a 14, de
modo que sin tratar la rotulacion el conteo se duplicaba.

Uso como modulo:
    from quitar_rotulacion import limpiar_rotulacion
    limpio, mascara_tinta = limpiar_rotulacion(crop, plate_mask, hue_agar)

Uso directo, para inspeccionar la mascara y el resultado:
    python scripts/quitar_rotulacion.py images/mis_fotos_lote3/RC73-7.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

DESVIO_TONO = 15      # grados de tono que separan la tinta del agar
SAT_MINIMA = 45       # por debajo de esto el color no esta definido
DILATACION = 5        # ensancha la mascara para cubrir el halo del trazo
AREA_MINIMA = 40      # descarta motas sueltas que no son trazos


def mascara_rotulacion(crop_bgr, plate_mask, hue_agar,
                       desvio=DESVIO_TONO, sat_minima=SAT_MINIMA):
    """
    Marca los pixeles que pertenecen a la rotulacion.

    El criterio es el mismo que se valido para el filtro por region, aplicado
    aqui pixel a pixel: tono que se aparta del agar y saturacion suficiente para
    que ese tono signifique algo.
    """
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    tono = hsv[..., 0].astype(np.int16)
    sat = hsv[..., 1].astype(np.int16)

    # diferencia circular de tono, porque el matiz es un angulo y 179 esta al
    # lado de 0
    dif = np.abs(tono - int(round(hue_agar)))
    dif = np.minimum(dif, 180 - dif)

    tinta = ((dif > desvio) & (sat > sat_minima)).astype(np.uint8)
    tinta[plate_mask == 0] = 0

    # los trazos son continuos; se cierran huecos y se descartan motas sueltas
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tinta = cv2.morphologyEx(tinta, cv2.MORPH_CLOSE, k, iterations=2)

    n, etiquetas, stats, _ = cv2.connectedComponentsWithStats(tinta, 8)
    limpia = np.zeros_like(tinta)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= AREA_MINIMA:
            limpia[etiquetas == i] = 1

    # se ensancha para cubrir el borde difuso del trazo, que conserva algo de
    # color y bastaria para confundir al detector
    k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                   (2 * DILATACION + 1,) * 2)
    return cv2.dilate(limpia, k2)


def limpiar_rotulacion(crop_bgr, plate_mask, hue_agar, margen_borde=18,
                       **kwargs):
    """
    Devuelve la imagen con la rotulacion reconstruida, la mascara empleada y la
    mascara de placa recortada alli donde no se puede reconstruir.

    Se usa inpainting de Telea, que propaga el color del contorno hacia dentro
    de la zona marcada. Funciona bien mientras el trazo este rodeado de agar,
    que es liso y no tiene estructura que adivinar.

    El caso que no funciona es el trazo pegado al limite de la placa: alli el
    algoritmo no tiene vecindario valido del que copiar y deja muescas dentadas
    en el borde, que el detector confunde con colonias. Se observo en RC73-8,
    donde esas muescas anadian una decena de detecciones falsas.

    La solucion es no pretender reconstruir lo irreconstruible. La rotulacion
    que toca el borde se excluye del area analizada en lugar de rellenarse, de
    modo que ni se reconstruye mal ni se cuenta. El coste es perder una franja
    estrecha del borde, donde de todos modos la escritura impide ver si hay
    colonias.
    """
    mascara = mascara_rotulacion(crop_bgr, plate_mask, hue_agar, **kwargs)
    if mascara.sum() == 0:
        return crop_bgr.copy(), mascara, plate_mask.copy()

    # franja interior del borde de la placa, donde el inpainting no es fiable
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (2 * margen_borde + 1,) * 2)
    interior = cv2.erode((plate_mask > 0).astype(np.uint8), k)
    borde = ((plate_mask > 0).astype(np.uint8) - interior)

    # la tinta que toca el borde se excluye; la del interior se reconstruye
    tinta_borde = cv2.bitwise_and(mascara, borde)
    n, etiquetas, _, _ = cv2.connectedComponentsWithStats(mascara, 8)
    a_excluir = np.zeros_like(mascara)
    for i in range(1, n):
        comp = (etiquetas == i).astype(np.uint8)
        if cv2.bitwise_and(comp, tinta_borde).any():
            a_excluir = cv2.bitwise_or(a_excluir, comp)
    a_reconstruir = cv2.bitwise_and(mascara, 1 - a_excluir)

    limpio = crop_bgr.copy()
    if a_reconstruir.any():
        limpio = cv2.inpaint(limpio, a_reconstruir, inpaintRadius=7,
                             flags=cv2.INPAINT_TELEA)

    mascara_util = plate_mask.copy()
    mascara_util[a_excluir > 0] = 0
    limpio[mascara_util == 0] = 0
    return limpio, mascara, mascara_util


def _inspeccionar(ruta):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from contar_mis_fotos import detect_plate, crop_plate
    from autocalibrar import calibrar

    img = cv2.imread(str(ruta))
    cx, cy, r = detect_plate(img)
    crop, mask = crop_plate(img, cx, cy, r)
    par = calibrar(crop, mask)
    limpio, mascara, util = limpiar_rotulacion(crop, mask, par['hue_agar'])

    cubierto = 100.0 * mascara.sum() / max(1, (mask > 0).sum())
    print(f'{Path(ruta).name}: tono del agar {par["hue_agar"]:.0f}, '
          f'{cubierto:.1f}% de la placa marcado como rotulacion')

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(f'{Path(ruta).name}: eliminación de la rotulación',
                 fontsize=13, fontweight='bold')
    axes[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Placa recortada', fontsize=11)
    axes[1].imshow(mascara, cmap='gray')
    axes[1].set_title(f'Máscara de rotulación\n{cubierto:.1f}% de la placa',
                      fontsize=11)
    axes[2].imshow(cv2.cvtColor(limpio, cv2.COLOR_BGR2RGB))
    axes[2].set_title('Reconstruida, entrada al detector', fontsize=11)
    for a in axes:
        a.axis('off')
    plt.tight_layout()
    salida = Path('results/colonias/experimentos/19_quitar_rotulacion')
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / f'{Path(ruta).stem}_limpieza.png'
    plt.savefig(destino, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'  {destino}')


if __name__ == '__main__':
    rutas = sys.argv[1:] or ['images/mis_fotos_lote3/RC73-7.jpg']
    for r in rutas:
        _inspeccionar(r)

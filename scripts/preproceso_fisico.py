"""
Preprocesamiento fundado en la fisica de la transiluminacion.

El preprocesamiento actual (flat_field en contar_mis_fotos.py) divide la imagen
por un desenfoque gaussiano de si misma. Es simple y funciono, pero tiene un
defecto conceptual: el desenfoque incluye a las propias colonias, de modo que en
una zona poblada el fondo estimado sale mas oscuro de lo que es y las colonias
de esa zona pierden contraste. Dicho de otro modo, el metodo confunde la
iluminacion con el contenido.

Este modulo implementa dos correcciones con base fisica.

1. Estimacion del fondo por morfologia

   En una placa iluminada por detras, las colonias son objetos oscuros pequenos
   sobre un fondo claro y suave. Una apertura morfologica en escala de grises con
   un elemento estructurante mayor que la colonia mas grande **elimina** esos
   objetos oscuros y deja solo la iluminacion. Es la forma estandar de separar
   fondo de contenido cuando se conoce el tamano maximo del contenido, y a
   diferencia del desenfoque no se contamina con las colonias.

   Despues se suaviza el resultado, porque la apertura deja escalones.

2. Densidad optica en vez de intensidad

   La ley de Beer-Lambert dice que la luz que atraviesa un medio absorbente cae
   de forma exponencial con la cantidad de material: I = I0 * exp(-k*c*d). Una
   colonia sobre un transiluminador es exactamente eso, una capa de biomasa que
   atenua la luz de fondo.

   Tomando el logaritmo, la densidad optica OD = -log(I / I0) resulta
   **proporcional a la biomasa**, mientras que la intensidad no lo es. Esto
   importa porque la transformacion linealiza el problema: una colonia con la
   misma biomasa produce el mismo valor de OD este donde este, aunque la
   iluminacion de fondo I0 varie por la placa. La correccion de iluminacion y la
   linealizacion se hacen entonces en un solo paso, que es fisicamente correcto,
   en lugar de una division seguida de un estiramiento arbitrario.

Uso como modulo:
    from preproceso_fisico import densidad_optica, fondo_morfologico

Uso directo, para comparar visualmente los metodos:
    python scripts/preproceso_fisico.py images/mis_fotos_lote3/RC73-1.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# Radio del elemento estructurante, en pixeles del recorte de 1200 px. Debe
# superar al radio de la colonia mas grande, porque la apertura solo borra lo
# que cabe dentro de el; si se queda corto, las colonias grandes se quedan
# impresas en el fondo y luego se cancelan a si mismas al dividir.
#
# Elegido midiendo la rugosidad del fondo estimado, es decir cuanto se aparta de
# ser liso, que es el residuo de las colonias que la apertura no elimino:
#
#   radio  55 -> rugosidad 0.103, la colonia filamentosa sigue impresa
#   radio  80 -> rugosidad 0.039
#   radio 110 -> rugosidad 0.018, fondo liso
#   radio 150 -> rugosidad 0.013, sin mejora apreciable
#
# Se toma 110, donde la curva se aplana. Un valor mucho mayor empezaria a
# perder la variacion real de iluminacion, que es lo que se quiere estimar.
RADIO_MAX_COLONIA = 110


def fondo_morfologico(gris, plate_mask, radio=RADIO_MAX_COLONIA):
    """
    Estima la iluminacion de fondo sin contaminarse con las colonias.

    Una apertura en escala de grises elimina los objetos oscuros menores que el
    elemento estructurante. Como las colonias son oscuras y acotadas en tamano,
    y la iluminacion varia de forma suave a lo largo de toda la placa, la
    apertura conserva la segunda y borra las primeras.
    """
    trabajo = gris.astype(np.float32)

    # fuera de la placa no hay informacion util; se rellena con la mediana
    # interior para que el elemento estructurante no arrastre el negro del borde
    dentro = plate_mask > 0
    if dentro.sum() == 0:
        return cv2.GaussianBlur(trabajo, (0, 0), radio)
    trabajo[~dentro] = float(np.median(trabajo[dentro]))

    k = 2 * radio + 1
    elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # apertura: erosion seguida de dilatacion. Sobre objetos oscuros equivale a
    # un cierre, asi que se opera sobre la imagen invertida
    fondo = 255.0 - cv2.morphologyEx(255.0 - trabajo, cv2.MORPH_OPEN, elem)

    # la apertura deja escalones; se suavizan sin volver a meter las colonias,
    # porque ya no estan
    fondo = cv2.GaussianBlur(fondo, (0, 0), radio * 0.6)
    return fondo


def densidad_optica(crop_bgr, plate_mask, radio=RADIO_MAX_COLONIA,
                    percentil_alto=99.5):
    """
    Devuelve la imagen convertida a densidad optica, en 8 bits para el modelo.

    OD = -log10(I / I0), con I0 el fondo estimado. El resultado es proporcional
    a la biomasa que atraviesa la luz, de modo que una colonia da el mismo valor
    en el centro de la placa que en la periferia, aunque alli llegue menos luz.

    Se devuelve invertida, con las colonias oscuras sobre fondo claro, porque es
    la convencion con la que se calibraron los filtros del pipeline y la que
    espera el modelo.
    """
    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fondo = fondo_morfologico(gris, plate_mask, radio)

    # se evita dividir por cero y tomar logaritmo de valores no positivos
    razon = np.clip(gris, 1.0, None) / np.clip(fondo, 1.0, None)
    od = -np.log10(np.clip(razon, 1e-4, 1.0))     # 0 donde no hay absorcion

    dentro = plate_mask > 0
    if dentro.sum() == 0:
        return crop_bgr
    alto = float(np.percentile(od[dentro], percentil_alto))
    if alto <= 0:
        alto = 1e-3
    od = np.clip(od / alto, 0, 1)

    # a 8 bits e invertida: colonias oscuras sobre agar claro
    salida = ((1.0 - od) * 255).astype(np.uint8)
    salida[~dentro] = 0
    return cv2.cvtColor(salida, cv2.COLOR_GRAY2BGR)


def _comparar(ruta):
    """Figura comparando el preprocesamiento actual con el propuesto."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from contar_mis_fotos import detect_plate, crop_plate, flat_field

    img = cv2.imread(str(ruta))
    cx, cy, r = detect_plate(img)
    crop, mask = crop_plate(img, cx, cy, r)

    gris = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fondo_gauss = cv2.GaussianBlur(gris, (int(max(gris.shape) * 0.15) | 1,) * 2, 0)
    fondo_morf = fondo_morfologico(gris, mask)

    actual = flat_field(crop, mask)
    propuesto = densidad_optica(crop, mask)

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle(f'{Path(ruta).name}: preprocesamiento actual frente al '
                 f'fundado en la fisica', fontsize=13, fontweight='bold')

    axes[0, 0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Placa recortada', fontsize=11)
    axes[0, 1].imshow(fondo_gauss, cmap='gray')
    axes[0, 1].set_title('Fondo por desenfoque\n(se contamina con las colonias)',
                         fontsize=11)
    axes[0, 2].imshow(cv2.cvtColor(actual, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Actual: division y CLAHE', fontsize=11)

    dentro = mask > 0
    diferencia = np.zeros_like(gris)
    diferencia[dentro] = (fondo_gauss - fondo_morf)[dentro]
    axes[1, 0].imshow(diferencia, cmap='coolwarm')
    axes[1, 0].set_title('Diferencia entre ambos fondos\n(lo que el desenfoque '
                         'se come)', fontsize=11)
    axes[1, 1].imshow(fondo_morf, cmap='gray')
    axes[1, 1].set_title('Fondo por morfología\n(solo iluminación)', fontsize=11)
    axes[1, 2].imshow(cv2.cvtColor(propuesto, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Propuesto: densidad óptica', fontsize=11)

    for a in axes.ravel():
        a.axis('off')
    plt.tight_layout()
    salida = Path('results/colonias/experimentos/18_preproceso_fisico')
    salida.mkdir(parents=True, exist_ok=True)
    destino = salida / f'{Path(ruta).stem}_comparacion.png'
    plt.savefig(destino, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Guardado {destino}')


if __name__ == '__main__':
    ruta = sys.argv[1] if len(sys.argv) > 1 else \
        'images/mis_fotos_lote3/RC73-1.jpg'
    _comparar(ruta)

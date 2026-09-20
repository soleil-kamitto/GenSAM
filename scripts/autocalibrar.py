"""
Auto-calibracion de los parametros del pipeline a partir de la propia imagen.

Problema que resuelve. El sistema esta pensado para laboratorios sin
presupuesto, donde cada uno usara un telefono distinto y una iluminacion
distinta. Pero se comprobo que tres parametros calibrados en un montaje fallan
en otro:

  el estimador de densidad, ajustado con las placas dobles, asigna 10 colonias a
  una placa que tiene 352 cuando se le pasan fotografias con contraluz
  el filtro de color, ajustado con el primer lote, queda al borde de descartar
  colonias reales en el segundo, porque el tono del agar paso de 43 a 56
  el umbral de deteccion, ajustado con las fotografias de contraluz, produce un
  sesgo de mas ocho colonias por placa en el conjunto de placas dobles

Los tres fallos comparten causa: son constantes fijadas para unas condiciones
concretas. Este modulo las sustituye por valores derivados de propiedades
medibles de cada fotografia, de modo que un laboratorio nuevo no tenga que
recalibrar nada.

Propiedades medidas sobre las tres colecciones disponibles, que definen los
rangos de referencia:

  conjunto           contraste  gradiente   tono   ruido
  placas dobles           43.5       99.5  110.0    29.8
  fotos con contraluz     35.6      122.4   45.0    17.9
  segundo lote            37.2      103.1   56.0    29.7

Uso como modulo:
    from autocalibrar import calibrar
    p = calibrar(crop, mascara)
    p['umbral'], p['hue_agar'], p['sat_minima']

Uso directo, para inspeccionar que valores saldrian en cada coleccion:
    python scripts/autocalibrar.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


def _moda_circular(tonos):
    """Casilla mas poblada de un histograma de tono suavizado en circulo."""
    t = np.asarray(tonos, dtype=np.int32).ravel()
    if t.size == 0:
        return 0.0
    h = np.bincount(t, minlength=180)[:180].astype(np.float32)
    # el suavizado evita que un pico repartido entre dos casillas vecinas
    # pierda frente a otro mas estrecho
    nucleo = np.array([1, 2, 3, 2, 1], dtype=np.float32)
    nucleo /= nucleo.sum()
    suave = np.convolve(np.concatenate([h[-2:], h, h[:2]]), nucleo, 'valid')
    return float(np.argmax(suave))


def tono_dominante(tonos, saturaciones=None, valores=None):
    """
    Tono del agar, tomado sobre el circulo y solo donde el tono significa algo.

    La mediana no sirve para un angulo. Si el agar reparte sus tonos entre 175 y
    5, cruzando la costura de la escala, la mediana cae cerca de 90, que es un
    verde inexistente en la imagen. Se toma en su lugar la moda de un histograma
    circular, que ademas es mas robusta, porque el agar ocupa la mayor parte del
    disco y domina el reparto aunque haya colonias y rotulacion.

    Queda un problema, y hay dos maneras de equivocarse al resolverlo, las dos
    comprobadas sobre las placas propias.

    En un pixel gris o negro el tono no significa nada y la conversion lo deja
    en cero, de modo que esos pixeles se amontonan en la primera casilla y
    pueden ganarla sin que haya ningun rojo en la imagen. Medido: en una placa
    con una zona oscura amplia la moda daba 0 frente a 46 de la mediana.

    Pesar cada pixel por su saturacion corrige ese caso pero abre el contrario,
    porque una region pequena e intensamente coloreada le gana entonces a un
    agar grande y palido. Medido: en una placa doble con contraluz la moda
    ponderada daba 21, un naranja, frente a 87 de la mediana.

    Lo que funciona es descartar los pixeles donde el tono no esta definido y
    contar por area los que quedan, de modo que gane la superficie mayor y no la
    mas vistosa. El piso de saturacion se toma de la propia imagen, porque una
    fotografia palida entera no debe quedarse sin pixeles validos.

    Comprobado sobre veinte placas de cuatro procedencias: coincide con la
    mediana dentro de un grado en todas las placas propias, arregla el caso de
    la zona oscura, y sobre los agares de sangre de ADBC, cuyo tono cae en 177 a
    179 justo en la costura, devuelve el valor correcto donde la mediana no
    podria. Dentro de un mismo montaje resulta ademas mas consistente que la
    mediana, con ocho grados de dispersion frente a veintitres.
    """
    t = np.asarray(tonos, dtype=np.int32).ravel()
    if t.size == 0:
        return 0.0
    if saturaciones is None:
        return _moda_circular(t)

    s = np.asarray(saturaciones, dtype=np.float32).ravel()
    v = (np.asarray(valores, dtype=np.float32).ravel()
         if valores is not None else np.full_like(s, 255.0))

    piso = max(20.0, float(np.median(s)) * 0.5)
    definido = (s >= piso) & (v >= 25.0)
    if definido.sum() < max(100, int(0.02 * t.size)):
        definido = s >= 20.0          # imagen palida, se relaja el criterio
    if definido.sum() < 100:
        return _moda_circular(t)      # sin color utilizable, se usa todo
    return _moda_circular(t[definido])


def medir_condiciones(crop_bgr, plate_mask):
    """
    Describe las condiciones de captura de una fotografia de placa.

    Todas las medidas se toman dentro del disco de la placa, porque el fondo de
    la escena varia con el encuadre y no dice nada sobre la imagen que importa.
    """
    gris = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    dentro = plate_mask > 0
    if dentro.sum() < 100:
        return None

    # el fondo es la imagen sin su estructura fina, o sea el agar y su
    # iluminacion; la diferencia con la imagen es la estructura, o sea las
    # colonias mas el ruido
    k = int(max(gris.shape) * 0.15) | 1
    fondo = cv2.GaussianBlur(gris, (k, k), 0)
    detalle = gris - fondo

    return {
        'contraste': float(gris[dentro].std()),
        'gradiente': float(np.percentile(fondo[dentro], 95)
                           - np.percentile(fondo[dentro], 5)),
        'hue_agar': tono_dominante(hsv[..., 0][dentro],
                                   hsv[..., 1][dentro],
                                   hsv[..., 2][dentro]),
        'sat_agar': float(np.median(hsv[..., 1][dentro])),
        'brillo': float(np.median(gris[dentro])),
        'ruido': float(detalle[dentro].std()),
    }


def calibrar(crop_bgr, plate_mask):
    """
    Devuelve los parametros del pipeline ajustados a esta fotografia.

    Las reglas se apoyan en relaciones fisicas, no en valores memorizados de un
    conjunto concreto, que es lo que hacia fallar a las versiones anteriores.
    """
    c = medir_condiciones(crop_bgr, plate_mask)
    if c is None:
        return {'umbral': 0.65, 'hue_agar': 45.0, 'sat_minima': 40.0,
                'aplicar_flat': True, 'condiciones': None}

    # Umbral de deteccion. Se apoya en la magnitud del detalle, es decir cuanto
    # se aparta la imagen de su propio fondo suavizado. Cuando las colonias
    # destacan con nitidez ese valor es alto y el detector puede ser exigente
    # sin perderlas; cuando son palidas y se confunden con el agar el valor baja
    # y hace falta bajar tambien el umbral.
    #
    # La recta se apoya en los dos montajes con resultado medido: el de placas
    # dobles, con detalle cercano a 30, necesita 0.80, y el de contraluz, con
    # detalle cercano a 18, necesita 0.40.
    #
    # ADVERTENCIA. Dos puntos no bastan para fijar una regla general, y esto es
    # justo el error que el modulo intenta corregir. La recta debe validarse con
    # fotografias de otros laboratorios antes de darla por buena; mientras
    # tanto es una hipotesis operativa, no un resultado.
    umbral = float(np.clip(0.40 + (c['ruido'] - 18.0) * 0.0333, 0.35, 0.80))

    # Correccion de iluminacion. Solo tiene sentido si hay un gradiente que
    # corregir. Con luz uniforme no corrige nada y realza la textura del agar,
    # lo que produce falsos positivos.
    aplicar_flat = c['gradiente'] > 110.0

    # Filtro de rotulacion. En vez de un rango fijo de tono, se toma el del agar
    # de esta misma fotografia y se rechaza lo que se aparta de el.
    #
    # La saturacion actua solo como salvaguarda, para no descartar regiones
    # grises o sin color definido. Una version anterior exigia 1.5 veces la
    # saturacion del agar, y fallaba: en fotografias donde el propio agar es
    # saturado el umbral se iba por encima de 80 y dejaba pasar la rotulacion.
    # Medido sobre RC73-7, las colonias se agrupan en tono 41 a 45 y la tinta en
    # 67 a 100, de modo que el tono separa por si solo y la saturacion no debe
    # estorbar esa separacion.
    sat_minima = max(30.0, c['sat_agar'] * 0.9)

    return {
        'umbral': umbral,
        'hue_agar': c['hue_agar'],
        'sat_minima': sat_minima,
        'aplicar_flat': aplicar_flat,
        'condiciones': c,
    }


def desvio_circular(tonos, referencia):
    """
    Distancia de cada tono a la referencia, contada sobre el circulo.

    El matiz es un angulo, de modo que en la escala de OpenCV, de 0 a 179, el
    valor 179 esta pegado al 0 y no a 90. Restar sin mas es correcto mientras el
    agar quede lejos de esa costura, y falla por completo cuando no.

    El fallo se encontro al pasar el pipeline a placas ajenas. Los agares de
    sangre tienen el tono del agar en 178 o en 6, es decir justo en la costura,
    y una colonia de tono 2 dista 4 grados del agar de tono 178 pero la resta
    directa da 176. El filtro la tomaba por rotulacion y la descartaba. En las
    fotografias propias esto nunca se manifesto porque su agar esta en 41 a 56,
    lejos de la costura, que es un buen ejemplo de error latente que solo
    aparece al cambiar de laboratorio.
    """
    d = (np.asarray(tonos, dtype=np.float32) - float(referencia) + 90.0) \
        % 180.0 - 90.0
    return np.abs(d)


def es_tinta(hsv, label_mask, label, hue_agar, sat_minima, desvio=18.0):
    """
    Decide si una region corresponde a rotulacion con marcador.

    Usa dos estadisticos del tono, no solo la mediana, porque una region que
    monta a medias sobre la escritura tiene mediana parecida a la del agar y aun
    asi es mayormente tinta. Medido sobre RC73-8, esas regiones mixtas dan
    mediana 44 con percentil 90 en 86, mientras que una colonia real da mediana
    41 con percentil 90 en 42, es decir ambos valores juntos.

    Se rechaza entonces cuando la desviacion tipica de la region se aparta del
    agar, que es el caso de la tinta limpia, o cuando lo hace su decil superior,
    que es el caso de la region contaminada. Una colonia que toque la escritura
    se descarta tambien, lo cual es razonable porque en ese caso no se puede
    separar una de otra.

    Ambos estadisticos se calculan sobre la distancia circular, por el motivo
    explicado en desvio_circular.
    """
    region = label_mask == label
    s = float(np.median(hsv[..., 1][region]))
    if s < sat_minima:
        return False
    d = desvio_circular(hsv[..., 0][region], hue_agar)
    return float(np.median(d)) > desvio or float(np.percentile(d, 90)) > desvio


def _inspeccionar():
    """Muestra que parametros saldrian para una imagen de cada coleccion."""
    from contar_mis_fotos import detect_plate, crop_plate

    muestras = [
        ('placas dobles', 'images/placas/actinomicetos_1.jpeg'),
        ('contraluz lote 1', 'images/mis_fotos/MC73-A.jpg'),
        ('contraluz lote 2', 'images/mis_fotos_lote2/NRC73-A-2.5-.jpg'),
    ]
    print(f'{"coleccion":<18} {"umbral":>7} {"flat":>6} {"hue agar":>9} '
          f'{"sat min":>8} {"gradiente":>10} {"relieve":>8}')
    print('-' * 72)
    for nombre, ruta in muestras:
        p = Path(ruta)
        if not p.exists():
            print(f'{nombre:<18} no encontrada')
            continue
        img = cv2.imread(str(p))
        cx, cy, r = detect_plate(img)
        crop, mask = crop_plate(img, cx, cy, r)
        par = calibrar(crop, mask)
        c = par['condiciones']
        relieve = c['ruido'] / (c['contraste'] + 1e-6)
        print(f'{nombre:<18} {par["umbral"]:>7.2f} '
              f'{"si" if par["aplicar_flat"] else "no":>6} '
              f'{par["hue_agar"]:>9.1f} {par["sat_minima"]:>8.1f} '
              f'{c["gradiente"]:>10.1f} {relieve:>8.3f}')
    print()
    print('Referencia de lo que cada coleccion necesita segun los experimentos:')
    print('  placas dobles      umbral alto (0.80), sin flat-field')
    print('  contraluz          umbral bajo (0.40), con flat-field')


if __name__ == '__main__':
    _inspeccionar()

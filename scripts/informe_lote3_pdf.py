"""
Genera el informe del tercer lote en PDF, en blanco y negro.

Contenido. Que se midio, que resultados salieron, que intentos se hicieron y
cuales fallaron, y el recorrido completo de cada fotografia desde la imagen
original hasta el conteo, una pagina por placa.

Por que en blanco y negro. El informe esta pensado para imprimirse y para
proyectarse, y ambos escenarios degradan el color de forma impredecible. Cuando
hay que distinguir dos cosas se usan formas distintas, un circulo y un aspa, en
lugar de dos colores, de modo que la distincion sobrevive a una fotocopia.

Requiere que la corrida haya guardado los pasos intermedios, es decir los
archivos <placa>_pasos.npz que escribe scripts/contar_auto.py.

Uso:
    python scripts/informe_lote3_pdf.py
"""
import sys
import textwrap
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle

sys.path.insert(0, str(Path(__file__).parent))
from comparar_espacial import emparejar

FOTOS = Path('images/mis_fotos_lote3')
PASOS = Path('results/colonias/mis_fotos_lote3_auto_shrink100')
BASE = Path('results/colonias/mis_fotos_lote3_auto')
GT = Path('results/ground_truth_mis_fotos_lote3')
SALIDA = Path('docs/informe_lote3.pdf')

A4 = (8.27, 11.69)
MARGEN = 0.09           # fraccion de la pagina
GRIS = '0.35'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif'],
    'text.color': 'black',
    'axes.edgecolor': 'black',
    'pdf.fonttype': 42,
})

PLACAS = [f'RC73-{i}' for i in range(1, 11)]
MANUAL = dict(zip(PLACAS, [31, 40, 33, 29, 37, 21, 16, 10, 23, 25]))


# ── utilidades de texto ──────────────────────────────────────────────────────
class Pagina:
    """Coloca texto de arriba abajo llevando la cuenta de la altura usada."""

    def __init__(self, pdf, titulo=None):
        self.fig = plt.figure(figsize=A4)
        self.pdf = pdf
        self.y = 1 - MARGEN
        if titulo:
            self.encabezado(titulo)

    def encabezado(self, texto):
        self.fig.text(MARGEN, self.y, texto, size=12, weight='bold', va='top')
        self.y -= 0.028
        self.fig.add_artist(plt.Line2D([MARGEN, 1 - MARGEN], [self.y, self.y],
                                       color='black', lw=0.8))
        self.y -= 0.022

    def apartado(self, texto):
        self.y -= 0.012
        self.fig.text(MARGEN, self.y, texto, size=10, weight='bold', va='top')
        self.y -= 0.024

    def parrafo(self, texto, ancho=96, salto=0.0155):
        for linea in textwrap.wrap(' '.join(texto.split()), ancho):
            self.fig.text(MARGEN, self.y, linea, size=8.6, va='top')
            self.y -= salto
        self.y -= 0.009

    def tabla(self, cabecera, filas, columnas, negritas=()):
        """
        columnas son desplazamientos desde el margen, no anchos de columna.
        Con anchos acumulados, una primera columna de ancho cero colocaba la
        segunda encima de la primera, y las cifras se imprimian sobre las
        etiquetas.
        """
        pos = [MARGEN + c for c in columnas]
        for x, c in zip(pos, cabecera):
            self.fig.text(x, self.y, c, size=8, weight='bold',
                          va='top', ha='left')
        self.y -= 0.016
        self.fig.add_artist(plt.Line2D([MARGEN, 1 - MARGEN], [self.y + 0.004,
                                       self.y + 0.004], color='black', lw=0.6))
        self.y -= 0.006
        for i, fila in enumerate(filas):
            peso = 'bold' if i in negritas else 'normal'
            for x, celda in zip(pos, fila):
                self.fig.text(x, self.y, str(celda), size=8.3, va='top',
                              weight=peso)
            self.y -= 0.0165
        self.y -= 0.012

    def nota(self, texto, ancho=104):
        self.y -= 0.006
        for linea in textwrap.wrap(' '.join(texto.split()), ancho):
            self.fig.text(MARGEN, self.y, linea, size=7.6, va='top',
                          color=GRIS, style='italic')
            self.y -= 0.0135
        self.y -= 0.008

    def cerrar(self, pie=None):
        if pie:
            self.fig.text(0.5, MARGEN * 0.55, pie, size=7.2, color=GRIS,
                          ha='center')
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


# ── paginas de texto ─────────────────────────────────────────────────────────
def portada(pdf):
    p = Pagina(pdf)
    p.fig.text(MARGEN, 1 - MARGEN, 'Conteo automático de colonias',
               size=15, weight='bold', va='top')
    p.y = 1 - MARGEN - 0.032
    p.fig.text(MARGEN, p.y, 'Lote RC73, diez placas de actinomicetos',
               size=11, va='top', color=GRIS)
    p.y -= 0.05
    p.fig.add_artist(plt.Line2D([MARGEN, 1 - MARGEN], [p.y, p.y],
                                color='black', lw=0.8))
    p.y -= 0.03

    p.parrafo('Este informe explica qué tan bien cuenta colonias el sistema '
              'automático sobre las diez placas del lote RC73, por qué se '
              'equivoca cuando se equivoca, y qué se probó para mejorarlo. '
              'Incluye el recorrido completo de cada fotografía, paso por paso, '
              'desde la imagen que salió del teléfono hasta el número final.')

    p.apartado('Qué se comparó')
    p.parrafo('La investigadora contó las colonias dos veces. Primero mirando '
              'las placas en la mano, y después sobre las mismas fotografías, '
              'marcando cada colonia con un clic que guarda su posición. Las '
              'dos cuentas dieron 265 colonias, exactamente el mismo número en '
              'las diez placas. Eso significa que la fotografía del teléfono no '
              'pierde colonias, y que cualquier diferencia que aparezca después '
              'es del programa y no de la cámara.')

    p.apartado('Los números')
    p.tabla(['', 'Mirando el 92 %', 'Mirando toda la placa'],
            [['Colonias reales', '265', '265'],
             ['Colonias contadas', '194', '266'],
             ['Acertó', '184', '241'],
             ['No vio', '81', '24'],
             ['Contó de más', '10', '25'],
             ['De las reales, cuántas vio', '69,4 %', '90,9 %'],
             ['De lo que contó, cuánto era real', '94,8 %', '90,6 %'],
             ['Error medio por placa', '7,1 colonias', '2,3 colonias']],
            [0.0, 0.40, 0.62], negritas=(5, 6))

    p.parrafo('La fila más importante es la penúltima. Cuando el sistema dice '
              'que algo es una colonia, acierta el 94,8 % de las veces. El '
              'problema no es que invente colonias, es que no las ve. En cuatro '
              'de las diez placas, todo lo que propuso era una colonia real.')

    p.nota('El conteo se hizo primero a ciegas, sin que el sistema ni quien lo '
           'ajustaba conocieran el resultado manual. Esa condición corresponde a '
           'la columna del 92 %. La columna de la placa entera se obtuvo después, '
           'ya con el conteo de referencia disponible.')
    p.cerrar('Página 1')


def por_que_falla(pdf):
    p = Pagina(pdf, 'Por qué no veía un tercio de las colonias')

    p.parrafo('Como el conteo manual guarda la posición de cada colonia, se '
              'pudo mirar dónde estaban exactamente las que el sistema no '
              'encontró. El resultado no deja lugar a dudas.')

    p.apartado('Dónde estaba cada colonia')
    p.tabla(['Distancia desde el centro', 'Las vio', 'No las vio'],
            [['Del centro hasta el 80 % del radio', '150', '4'],
             ['Entre el 80 % y el 90 %', '31', '5'],
             ['Entre el 90 % y el borde', '3', '64'],
             ['Justo en el borde o fuera', '0', '8']],
            [0.0, 0.52, 0.66], negritas=(2,))

    p.parrafo('Hasta el 90 % del radio el sistema encuentra 181 de 190 '
              'colonias. Más allá encuentra 3 de 75. El cambio no es suave, es '
              'un corte, y cae justo donde el programa deja de mirar.')

    p.apartado('La causa')
    p.parrafo('Antes de analizar la placa, el programa la recorta al 92 % de su '
              'radio y descarta el anillo de fuera, que es el 15 % de la '
              'superficie. Ese recorte se puso por un motivo real: en el borde '
              'están el menisco del agar, la pared del plástico y las gotas de '
              'condensación, y esas gotas son redondas y el detector las '
              'confunde con colonias.')
    p.parrafo('El problema es que en este lote las colonias crecieron sobre todo '
              'en el borde. La mitad de las colonias que el sistema no vio están '
              'entre el 90 % y el 100 % del radio. En total, 69 de las 81 '
              'colonias perdidas estaban fuera de la zona que el programa llega '
              'a mirar, y solo 12 se perdieron dentro.')

    p.apartado('Por qué el recorte estaba en 92 %')
    p.parrafo('Ese valor se eligió midiendo sobre un lote anterior de placas, '
              'donde recortar al 92 % dejaba dentro el 96 % de las colonias. En '
              'este lote deja dentro el 74 %. Es la misma investigadora, el '
              'mismo laboratorio y el mismo tipo de placa, y aun así el valor ya '
              'no sirve. Ese es el hallazgo de fondo del trabajo: los números '
              'que se ajustan mirando unas fotos concretas dejan de funcionar '
              'con otras, aunque parezcan muy parecidas.')
    p.cerrar('Página 2')


def intentos(pdf):
    p = Pagina(pdf, 'Qué se probó, y qué funcionó')

    p.parrafo('Se hicieron varios intentos antes de llegar al diagnóstico. '
              'Conviene dejarlos escritos, incluidos los que fallaron, porque '
              'cada uno acota lo que el sistema puede y no puede hacer.')

    p.apartado('1. Quitar la rotulación antes de analizar')
    p.parrafo('Las placas están rotuladas con marcador. En las más escritas, el '
              'detector proponía más trazos que colonias, hasta 28 frente a 14, '
              'de modo que el conteo se duplicaba. Se probaron cuatro filtros de '
              'color que descartaban los trazos después de detectarlos, y los '
              'cuatro fallaron por lo mismo: cuando una colonia toca un trazo, '
              'quedan unidos en una sola región y ya no se pueden separar. Lo '
              'que funcionó fue borrar la tinta antes de analizar y reconstruir '
              'el agar que hay debajo. Es legítimo porque la escritura está '
              'sobre el plástico y no en el cultivo. Resultado: funciona, y está '
              'en uso.')

    p.apartado('2. Separar colonias por su forma, sin usar el color')
    p.parrafo('El marcador negro no tiene color, así que un filtro por color no '
              'lo alcanza. Se probó distinguir los trazos por su forma, ya que '
              'un trazo es largo y estrecho mientras que una colonia es redonda. '
              'Sobre este lote separaba perfectamente. Al probarlo en placas muy '
              'pobladas y sin ninguna escritura, marcó el 70 % de las manchas '
              'oscuras, porque cuando las colonias se tocan forman cadenas tan '
              'alargadas como un trazo. Resultado: descartado.')

    p.apartado('3. Tres formas distintas de preparar la imagen')
    p.parrafo('Se contó el lote con tres preparaciones de fundamento distinto: '
              'la imagen directa, una transformación que mide cuánta luz absorbe '
              'cada punto, y el troceado de la placa en cuadrantes. Los totales '
              'fueron 194, 204 y 192 colonias. Las tres coinciden dentro de un '
              '6 %, lo que indica que la forma de preparar la imagen no es lo '
              'que decide el resultado. Resultado: útil para acotar el margen de '
              'error, pero no resuelve el problema.')

    p.apartado('4. Analizar la placa entera')
    p.parrafo('Quitar el recorte y mirar toda la placa sube las colonias '
              'encontradas de 184 a 241, y el error medio baja de 7,1 a 2,3 '
              'colonias por placa. A cambio, entran algunas gotas de '
              'condensación y el sistema pasa a contar de más 25 veces en lugar '
              'de 10. Aplicado al lote anterior, donde las colonias no están en '
              'el borde, este cambio empeora el resultado. Resultado: mejora '
              'mucho este lote y perjudica el otro, así que no se adopta como '
              'está.')

    p.apartado('5. Lo que viene')
    p.parrafo('La solución no es elegir mejor el porcentaje del recorte, sino '
              'dejar de usar la posición como criterio. Se analizará la placa '
              'entera y se decidirá por el brillo: las colonias de este lote son '
              'entre un 30 y un 45 % más oscuras que el agar, mientras que una '
              'gota de condensación es más clara que el fondo, porque refleja la '
              'luz en vez de absorberla. Eso se mide en cada fotografía y no '
              'depende de cómo se tomó. Está en desarrollo.')
    p.cerrar('Página 3')


def explicacion_proceso(pdf):
    p = Pagina(pdf, 'Qué le pasa a cada fotografía')
    p.parrafo('Las páginas siguientes muestran una placa por hoja, con los seis '
              'pasos que recorre la imagen. Esto es lo que hace cada uno.')

    pasos = [
        ('1. Fotografía original',
         'La imagen tal como sale del teléfono, sin tocar. Aquí la placa suele '
         'estar descentrada, con fondo alrededor y con la rotulación a la vista.'),
        ('2. Se localiza la placa',
         'El programa busca el círculo de la placa y recorta por ahí. Todo lo '
         'que quede fuera del círculo se pone a negro, para que el fondo de la '
         'mesa no cuente como parte del cultivo.'),
        ('3. Se prepara la imagen',
         'Se borra la rotulación y se reconstruye el agar que hay debajo. '
         'Cuando la iluminación es despareja, se corrige para que el borde no '
         'quede más oscuro que el centro. Esta es la imagen que recibe el '
         'modelo.'),
        ('4. El modelo propone regiones',
         'El modelo marca todas las manchas que le parecen un objeto. En este '
         'paso todavía no decide qué es una colonia, solo separa objetos del '
         'fondo. Cada mancha se dibuja en un gris distinto.'),
        ('5. Se descartan las que no son colonias',
         'Se eliminan las manchas demasiado pequeñas o demasiado grandes, las '
         'que tienen forma irregular y las que conservan color de marcador. Lo '
         'que sobrevive es el conteo del sistema, marcado con círculos.'),
        ('6. Se compara con el conteo manual',
         'Cada colonia marcada a mano se empareja con la detección más cercana. '
         'El círculo indica que el sistema la encontró, el aspa que se le pasó, '
         'y el círculo punteado que contó algo donde no había colonia.'),
    ]
    for titulo, texto in pasos:
        p.apartado(titulo)
        p.parrafo(texto)

    p.nota('El emparejamiento del paso 6 reparte las colonias de forma óptima en '
           'lugar de ir tomando la más cercana una por una, porque con colonias '
           'juntas ese atajo puede asignar mal la primera y arrastrar el error a '
           'las siguientes.')
    p.cerrar('Página 4')


# ── paginas de proceso, una por placa ────────────────────────────────────────
def gris(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def pagina_placa(pdf, nombre, numero):
    npz = PASOS / f'{nombre}_pasos.npz'
    det = PASOS / f'{nombre}_detecciones.csv'
    if not npz.exists():
        return False
    d = np.load(npz)
    crop, mplaca = d['recorte'], d['mascara_placa']
    entrada, etiquetas = d['entrada'], d['etiquetas']
    validas = set(int(x) for x in d['validas'])

    original = cv2.imread(str(FOTOS / f'{nombre}.jpg'))
    puntos = pd.read_csv(GT / f'{nombre}_puntos.csv')
    dets = pd.read_csv(det) if det.exists() else pd.DataFrame(columns=['x', 'y'])

    fig = plt.figure(figsize=A4)
    manual = MANUAL[nombre]
    sistema = len(dets)

    m = puntos[['x', 'y']].to_numpy(float)
    dd = dets[['x', 'y']].to_numpy(float) if len(dets) else np.empty((0, 2))
    rr = dets['radio_original'].to_numpy(float) if len(dets) else np.empty(0)
    par, per, sob = emparejar(m, dd, rr)

    fig.text(MARGEN, 1 - MARGEN, f'Placa {nombre}', size=12, weight='bold',
             va='top')
    fig.text(1 - MARGEN, 1 - MARGEN,
             f'conteo manual {manual}   ·   sistema {sistema}',
             size=9.5, va='top', ha='right')
    fig.add_artist(plt.Line2D([MARGEN, 1 - MARGEN], [1 - MARGEN - 0.022,
                              1 - MARGEN - 0.022], color='black', lw=0.8))

    paneles = []
    ancho, alto = 0.385, 0.255
    x0, x1 = MARGEN, MARGEN + 0.425
    ys = [0.665, 0.375, 0.085]
    for y in ys:
        paneles.append((x0, y)); paneles.append((x1, y))

    def eje(i):
        x, y = paneles[i]
        a = fig.add_axes([x, y, ancho, alto])
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_linewidth(0.6)
        return a

    def rotulo(a, texto):
        a.set_title(texto, size=8.6, pad=4, loc='left')

    a = eje(0); a.imshow(gris(original), cmap='gray', vmin=0, vmax=255)
    rotulo(a, '1. Fotografía original')

    a = eje(1); a.imshow(gris(crop), cmap='gray', vmin=0, vmax=255)
    rotulo(a, '2. Placa localizada y recortada')

    a = eje(2); a.imshow(gris(entrada), cmap='gray', vmin=0, vmax=255)
    rotulo(a, '3. Imagen preparada para el modelo')

    a = eje(3)
    vis = np.zeros(etiquetas.shape, dtype=float)
    ids = [i for i in np.unique(etiquetas) if i != 0]
    rng = np.random.default_rng(7)
    for i in ids:
        vis[etiquetas == i] = rng.uniform(0.35, 1.0)
    a.imshow(vis, cmap='gray', vmin=0, vmax=1)
    rotulo(a, f'4. Regiones propuestas ({len(ids)})')

    a = eje(4); a.imshow(gris(crop), cmap='gray', vmin=0, vmax=255)
    from skimage.measure import regionprops
    for pr in regionprops(etiquetas):
        if pr.label in validas:
            rad = max(np.sqrt(pr.area / np.pi) * 1.5, 7)
            a.add_patch(Circle((pr.centroid[1], pr.centroid[0]), rad,
                               fill=False, ec='black', lw=0.9))
    rotulo(a, f'5. Lo que contó como colonia ({sistema})')

    a = eje(5); a.imshow(gris(original), cmap='gray', vmin=0, vmax=255)
    for _, j in par:
        a.add_patch(Circle((dd[j, 0], dd[j, 1]), max(rr[j] * 1.5, 14),
                           fill=False, ec='black', lw=1.1))
    for i in per:
        a.plot(m[i, 0], m[i, 1], 'x', color='black', ms=6, mew=1.4)
    for j in sob:
        a.add_patch(Circle((dd[j, 0], dd[j, 1]), max(rr[j] * 1.5, 14),
                           fill=False, ec='black', lw=1.0, ls=(0, (2, 2))))
    rotulo(a, f'6. Acertó {len(par)} · no vio {len(per)} · de más {len(sob)}')

    fig.text(0.5, MARGEN * 0.55,
             f'Página {numero}  ·  círculo: encontrada  ×: no detectada  '
             'círculo punteado: contada de más',
             size=7.2, color=GRIS, ha='center')
    pdf.savefig(fig)
    plt.close(fig)
    return True


def main():
    faltan = [p for p in PLACAS if not (PASOS / f'{p}_pasos.npz').exists()]
    if faltan:
        print(f'Faltan los pasos intermedios de {len(faltan)} placas: '
              f'{", ".join(faltan)}')
        print('Se generan con: python scripts/contar_auto.py '
              'images/mis_fotos_lote3 --shrink 1.0')
        return

    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(SALIDA) as pdf:
        portada(pdf)
        por_que_falla(pdf)
        intentos(pdf)
        explicacion_proceso(pdf)
        n = 5
        for nombre in PLACAS:
            if pagina_placa(pdf, nombre, n):
                print(f'  {nombre}')
                n += 1
        info = pdf.infodict()
        info['Title'] = 'Conteo automático de colonias, lote RC73'
        info['Subject'] = 'Validación sobre diez placas de actinomicetos'

    print(f'\nInforme en {SALIDA}  ({SALIDA.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()

"""
Herramienta de conteo manual de colonias por clic, para construir el ground
truth de las placas propias (images/mis_fotos/).

A diferencia del ground truth anterior (un CSV escrito a mano con totales),
esta herramienta guarda las coordenadas de cada clic, lo que permite pausar y
retomar, corregir puntos y, mas adelante, evaluar localizacion ademas del
conteo total.

Uso:
    python scripts/conteo_manual.py [images/mis_fotos]

Controles:
    clic izquierdo   marcar una colonia
    rueda del mouse  zoom centrado en el cursor (no hace falta la lupa)
    clic derecho     borrar la marca mas cercana al cursor
    z                deshacer la ultima marca
    h                vista completa (reset del zoom)
    n / flecha der.  siguiente imagen (guarda automaticamente)
    p / flecha izq.  imagen anterior (guarda automaticamente)
    g                guardar ahora
    q                guardar y salir

Si el clic mueve la imagen en vez de marcar, es que la herramienta de
mover/lupa de la barra esta activa; desactivala con un clic en su icono.

Guarda:
    results/ground_truth_mis_fotos/<imagen>_puntos.csv     (x, y de cada marca)
    results/ground_truth_mis_fotos/ground_truth.csv        (resumen: imagen, conteo)
    results/ground_truth_mis_fotos/imagenes/<imagen>_gt.png (foto con las marcas)

La imagen con las marcas se guarda junto con el CSV cada vez que se guarda, de
modo que la evidencia visual siempre corresponde al conteo registrado.
Los puntos existentes se recargan al abrir, se puede retomar donde quedaste.
"""
import sys
import csv
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SUPPORTED = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
# La carpeta de salida se deriva del nombre de la carpeta de imagenes. Estaba
# fija, y con ella contar un lote nuevo sobrescribia el resumen del anterior,
# porque ground_truth.csv se reescribe entero con las placas de la carpeta
# actual. Para images/mis_fotos el resultado es la misma ruta de siempre, de
# modo que los conteos ya hechos siguen donde estaban.
OUT_DIR = Path('results/ground_truth_mis_fotos')
IMG_DIR = OUT_DIR / 'imagenes'


def _fijar_salida(images_dir):
    global OUT_DIR, IMG_DIR
    OUT_DIR = Path('results') / f'ground_truth_{Path(images_dir).name}'
    IMG_DIR = OUT_DIR / 'imagenes'


class Contador:
    def __init__(self, imagenes):
        self.imagenes = imagenes
        self.idx = 0
        self.puntos = {p.stem: self._cargar(p.stem) for p in imagenes}

        self.aviso = ''
        self.fig, self.ax = plt.subplots(figsize=(13, 9))
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.mostrar()

    # ── persistencia ─────────────────────────────────────────────────────────
    def _ruta_puntos(self, stem):
        return OUT_DIR / f'{stem}_puntos.csv'

    def _cargar(self, stem):
        ruta = self._ruta_puntos(stem)
        if not ruta.exists():
            return []
        with open(ruta, newline='') as f:
            return [(float(r['x']), float(r['y'])) for r in csv.DictReader(f)]

    def _ruta_imagen_gt(self, stem):
        return IMG_DIR / f'{stem}_gt.png'

    def guardar_imagen(self, stem):
        """
        Guarda la fotografia con las marcas del conteo superpuestas.

        Se genera en el momento del conteo, con las mismas coordenadas que se
        acaban de registrar, para que la evidencia visual y el CSV no puedan
        quedar desincronizados.
        """
        ruta_img = next((p for p in self.imagenes if p.stem == stem), None)
        if ruta_img is None:
            return
        pts = self.puntos[stem]

        img = np.asarray(Image.open(ruta_img))
        alto, ancho = img.shape[:2]
        fig = plt.figure(figsize=(ancho / 200, alto / 200), dpi=200)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(img)
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, '+', color='red', markersize=10,
                    markeredgewidth=1.8)
        ax.axis('off')
        IMG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(self._ruta_imagen_gt(stem), dpi=200)
        plt.close(fig)

    def guardar(self):
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for stem, pts in self.puntos.items():
            with open(self._ruta_puntos(stem), 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['x', 'y'])
                w.writerows(pts)
        with open(OUT_DIR / 'ground_truth.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['image', 'count'])
            for p in self.imagenes:
                w.writerow([p.stem, len(self.puntos[p.stem])])
        # solo la placa en curso, para no re-renderizar las 15 en cada guardado
        self.guardar_imagen(self.actual.stem)

    # ── interfaz ─────────────────────────────────────────────────────────────
    @property
    def actual(self):
        return self.imagenes[self.idx]

    def mostrar(self, mantener_vista=False):
        if mantener_vista:
            xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.clear()
        img = np.asarray(Image.open(self.actual))
        self.ax.imshow(img)
        pts = self.puntos[self.actual.stem]
        if pts:
            xs, ys = zip(*pts)
            self.ax.plot(xs, ys, '+', color='red', markersize=12,
                         markeredgewidth=2)
        aviso = f'\n>>> {self.aviso} <<<' if self.aviso else ''
        self.aviso = ''
        self.ax.set_title(
            f'[{self.idx + 1}/{len(self.imagenes)}]  {self.actual.name}   '
            f'conteo = {len(pts)}\n'
            'clic izq: marcar | rueda: zoom | clic der: borrar | z: deshacer | '
            'h: vista completa | n/p: cambiar | q: salir' + aviso,
            color='red' if aviso else 'black')
        self.ax.axis('off')
        if mantener_vista:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.fig.canvas.toolbar is not None and \
                self.fig.canvas.toolbar.mode != '':
            # una herramienta de la barra esta activa, avisar en vez de callar
            self.aviso = ('herramienta de mover/lupa activa, desactivala en '
                          'la barra para poder marcar')
            self.mostrar(mantener_vista=True)
            return
        pts = self.puntos[self.actual.stem]
        if event.button == 1:
            pts.append((event.xdata, event.ydata))
        elif event.button == 3 and pts:
            dists = [np.hypot(x - event.xdata, y - event.ydata)
                     for x, y in pts]
            pts.pop(int(np.argmin(dists)))
        self.mostrar(mantener_vista=True)

    def on_scroll(self, event):
        # zoom con la rueda, centrado en el cursor
        if event.inaxes != self.ax or event.xdata is None:
            return
        factor = 0.8 if event.button == 'up' else 1.25
        x, y = event.xdata, event.ydata
        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        self.ax.set_xlim(x - (x - x0) * factor, x + (x1 - x) * factor)
        self.ax.set_ylim(y - (y - y0) * factor, y + (y1 - y) * factor)
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        pts = self.puntos[self.actual.stem]
        if event.key == 'z' and pts:
            pts.pop()
            self.mostrar(mantener_vista=True)
            return
        elif event.key == 'h':
            self.mostrar()   # sin mantener_vista: resetea el zoom
            return
        elif event.key in ('n', 'right'):
            self.guardar()
            self.idx = (self.idx + 1) % len(self.imagenes)
        elif event.key in ('p', 'left'):
            self.guardar()
            self.idx = (self.idx - 1) % len(self.imagenes)
        elif event.key == 'g':
            self.guardar()
            print('Guardado.')
        elif event.key == 'q':
            self.guardar()
            plt.close(self.fig)
            return
        self.mostrar()


def main():
    images_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('images/mis_fotos')
    _fijar_salida(images_dir)
    imagenes = sorted(p for p in images_dir.iterdir()
                      if p.suffix.lower() in SUPPORTED)
    if not imagenes:
        print(f'No hay imagenes en {images_dir}')
        return

    print(f'{len(imagenes)} imagenes. Los puntos se guardan en {OUT_DIR}/')
    contador = Contador(imagenes)
    plt.show()

    contador.guardar()
    print('\nResumen final:')
    for p in imagenes:
        print(f'  {p.stem:<16} {len(contador.puntos[p.stem])}')
    print(f'\nGround truth: {OUT_DIR / "ground_truth.csv"}')


if __name__ == '__main__':
    main()

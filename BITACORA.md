# Bitácora del proyecto

Registro cronológico de avances, decisiones y resultados del sistema de conteo
automático de colonias de actinomicetos.

Cada entrada anota qué se hizo, qué se midió y qué se concluyó. Los resultados
negativos se conservan, porque suelen ser tan informativos como los positivos y
evitan repetir caminos ya explorados.

**Convención de métricas.** El error absoluto medio (MAE) se expresa en colonias
por placa. El acierto agregado se calcula como 1 menos el cociente entre el error
absoluto total y el total de colonias contadas a mano, de modo que no se
distorsiona con las placas de pocas colonias. Las placas con más de 250 colonias
se marcan como TNTC (*too numerous to count*), el límite convencional en
microbiología, y se reportan aparte.

---

## Fase 1. Conjunto de referencia, placas dobles

**Datos.** `images/placas`, 8 fotografías con dos placas cada una, 16 placas y
819 colonias contadas a mano. Iluminación directa, dos placas por encuadre.

**Hallazgo inicial.** CellSAM aplicado directamente sobre estas imágenes produce
**cero detecciones**. El modelo, entrenado con microscopía, no distingue las
colonias del agar sin normalización previa de contraste.

**Solución.** Preprocesamiento con CLAHE, que habilita la detección.

**Barrido del umbral de detección.** Se probaron ocho valores de
`bbox_threshold` entre 0.10 y 0.80
(`results/colonias/experimentos/05_bbox_sweep/`).

| Umbral | MAE |
|--------|-----|
| 0.10 | 22.56 |
| 0.40 | 12.75 |
| 0.60 | 8.50 |
| 0.80 | 7.06 |

**Selector adaptativo.** Se añadió un módulo que estima la densidad colonial por
visión clásica y elige el umbral en consecuencia. Resultado: **MAE 5.94**, mejor
marca del proyecto en esta fase, con un sesgo de −0.9 colonias por placa.

**Intentos de fine-tuning, ambos fallidos.**

| Variante | MAE |
|----------|-----|
| Ajuste del decodificador de máscaras | 12.75 |
| Ajuste de AnchorDETR sin normalización | 51.20 |
| **Modelo base sin ajustar** | **5.94** |

Ajustar el modelo con 16 placas lo empeora. La hipótesis es falta de datos, y es
lo que más adelante motiva la generación de placas sintéticas.

---

## Fase 2. Fotografías propias, primer lote

**Datos.** `images/mis_fotos`, 15 placas, una por imagen, capturadas con cámara
de teléfono sobre transiluminador. Series M1, M2, MC73 y RC73.

**Tres dificultades nuevas** respecto al conjunto anterior: gradiente de
iluminación por el contraluz, rotulación con marcador dentro del encuadre, y una
banda de artefactos en el borde donde se acumulan menisco y condensación.

**Problema de escala.** Las fotografías llegan a 4096 px, frente a los 1600 px
del conjunto anterior. La primera ejecución se colgó por presión de memoria. Se
resolvió reescalando el recorte de la placa a 1200 px, la escala con la que se
calibraron los filtros de área.

**Fallo con placas estériles.** El postprocesado de CellSAM lanza `ValueError` al
reducir un arreglo vacío cuando no detecta nada. Corregido capturando la
excepción.

**Corrección de iluminación de campo plano, el hallazgo decisivo.** Se estima el
fondo con un desenfoque gaussiano de núcleo igual al 15 % del lado y se divide la
imagen por él, seguido de CLAHE.

| Placa | Sin corrección, 0.65 | Sin corrección, 0.40 | Con corrección, 0.65 | Con corrección, 0.40 |
|-------|---------------------|---------------------|---------------------|---------------------|
| M2-A-3 | 0 | 0 | 15 | 29 |
| MC73-C | 1 | 3 | 4 | 4 |
| MC73-A, control | 13 | 13 | 13 | 13 |

Bajar el umbral por sí solo no recupera nada, de modo que el problema no era la
sensibilidad del detector sino el contraste de la imagen de entrada. El control
no se degrada.

**Filtro de rotulación.** Sobre una placa estéril, el marcador azul del borde
generaba 8 detecciones falsas. Midiendo el matiz de cada región aparece una
separación limpia: la tinta cae en 71 a 105 y las colonias en 39 a 44. Se
descarta por color.

**Resultado negativo útil.** Se sospechaba que el filtro de área mínima de 300 px²
eliminaba colonias pequeñas. La medición mostró que la región más pequeña
propuesta por el modelo tiene 648 px², muy por encima del umbral. Las colonias
que faltan no se filtran, es que el detector no las propone. El límite está en el
modelo, no en el postprocesado.

---

## Fase 3. Conteo manual de referencia

Se construyó una herramienta de anotación por clic
(`scripts/conteo_manual.py`) que registra las coordenadas de cada colonia, no
solo el total. Eso permite auditar el conteo y habilita evaluaciones de
localización.

**856 colonias marcadas a mano** en las 15 placas del primer lote.

---

## Fase 4. Calibración del recorte

Con las coordenadas del conteo manual se midió qué fracción de las colonias
reales queda dentro del área analizada según el radio de recorte.

| Recorte | Colonias dentro | Porcentaje |
|---------|-----------------|------------|
| 0.86 | 676 | 79.0 % |
| 0.90 | 761 | 88.9 % |
| **0.92** | **822** | **96.0 %** |
| 1.00 | 856 | 100 % |

El recorte al 86 % dejaba fuera el 21 % de las colonias reales.

**Comparación contra el conteo manual.**

| Recorte | MAE contable | MAE global | Acierto |
|---------|--------------|------------|---------|
| 0.86 | 5.46 | 15.40 | 73.0 % |
| **0.92** | **4.08** | **11.33** | **80.1 %** |
| 1.00 | 5.77 | 10.87 | sin medir |

**Por qué el radio completo es peor.** Se probó con un filtro de forma pensado
para descartar artefactos del borde, asumiendo que el menisco y la condensación
forman arcos alargados. La medición mostró que en estas fotografías la
condensación forma **gotas redondas**, indistinguibles de una colonia por su
forma. El filtro no las separa y la placa de control se rompió, pasando de 13
aciertos exactos a 16 detecciones.

---

## Fase 5. Segundo lote y fragilidad del color

**Datos.** `images/mis_fotos_lote2`, 4 placas NRC73, reservadas como **prueba
ciega**: no se usan para ajustar ningún parámetro ni se consulta su conteo
manual.

**Hallazgo.** El balance de color de la cámara cambió entre lotes.

| Lote | Matiz mediano del agar | Percentil 90 |
|------|------------------------|--------------|
| Primero | 43 a 48 | 46 |
| Segundo | 56 | 66 |

El filtro de tinta usa un umbral fijo en 60, así que en el segundo lote las
colonias quedan al borde de ser descartadas. Conservó el margen por poco.

**Contribución del filtro, medida** desactivándolo sobre las 15 placas
(`results/colonias/experimentos/14_tinta_adaptativa/`).

| Variante | MAE contable | Acierto |
|----------|--------------|---------|
| Sin filtro | 7.31 | 75.7 % |
| **Fijo** | **4.00** | **80.3 %** |
| Adaptativo al agar | 4.08 | 80.2 % |

El filtro aporta unos cuatro puntos y medio de acierto. La versión adaptativa
resulta equivalente sobre el primer lote, algo esperable porque el rango fijo se
calibró con esas mismas imágenes. Su ventaja está por comprobar en el segundo
lote.

---

## Fase 6. Estrategias descartadas

**Métodos clásicos** (`results/colonias/experimentos/12_benchmark/`).

| Método | MAE contable | Acierto |
|--------|--------------|---------|
| **CellSAM** | **4.15** | **80.0 %** |
| Watershed | 39.77 | 23.3 % |
| blob_doh | 66.31 | −14.2 % |
| Hough | 138.85 | −167.1 % |
| blob_log | 171.85 | −180.2 % |

Los métodos geométricos fracasan porque no distinguen colonia de textura. Sobre
M2-B, que tiene **cero colonias reales**, blob_log detecta 225 y Hough 299.

**Barrido de umbral y mosaico**
(`results/colonias/experimentos/15_mosaico_umbral/`).

| Estrategia | MAE contable | MAE global | Acierto |
|------------|--------------|------------|---------|
| **Entero 0.40** | **4.08** | 11.33 | 80.2 % |
| Entero 0.25 | 4.38 | **10.20** | **82.1 %** |
| Mosaico 2×2 con 0.25 | 14.15 | 18.13 | 68.3 % |

El mosaico queda descartado: sobredetecta gravemente en placas escasas, con 72
detecciones en una placa de cero colonias. El umbral 0.25 presenta un compromiso,
mejora el acierto agregado pero empeora el error placa a placa.

**Cellpose.** Inviable en el equipo disponible. Cinco horas y media de cómputo
sin completar una sola placa en CPU. Queda pendiente de evaluar con GPU, no
descartado.

---

## Fase 7. Comparación con el estado del arte

**Revisión de la literatura.** El conteo automático de colonias con aprendizaje
profundo es un campo maduro. Ya están publicados el uso de SAM sobre placas de
Petri, los modelos de fundación para colonias con pocos datos, el conteo desde
teléfono y varias variantes de YOLO. Existe incluso software comercial que se
anuncia para actinobacterias.

**Dataset AGAR.** 18.000 fotografías con 336.442 colonias anotadas, de cinco
especies, ninguna de ellas actinomiceto. La muestra libre de 40 imágenes está en
`images/agar_muestra`.

**Comparación morfológica, con la escala controlada**
(`results/colonias/experimentos/17_morfologia_normalizada/`).

| Descriptor | AGAR | Actinomicetos propios |
|------------|------|----------------------|
| Solidez | 0.834 | **0.884** |
| Circularidad | 0.507 | **0.647** |
| Exceso de perímetro | **1.698** | 1.439 |
| Nitidez del borde | 64.3 | **100.3** |
| Contraste | 0.238 | 0.264 |

**Este resultado contradice la premisa que se venía asumiendo.** Las colonias
propias miden como más compactas, más circulares y de borde más nítido que las de
AGAR, con significancia alta en los cinco descriptores. El argumento de que los
actinomicetos son un caso más difícil por su morfología filamentosa **no se
sostiene con estos datos**.

Una primera versión de esta medida, sin controlar la escala, exageraba las
diferencias; al normalizar el tamaño de cada colonia antes de medirla, el efecto
se atenuó pero mantuvo la dirección. Queda una salvedad sin resolver: las cajas
de AGAR pueden contener dos colonias que se tocan, lo que bajaría su solidez y
subiría su perímetro, y el conteo manual propio favorece las colonias bien
definidas.

**YOLOv8 entrenado con la muestra de AGAR**, evaluado sobre las fotografías
propias.

| | |
|---|---|
| mAP50 sobre el conjunto de prueba de AGAR | **0.726** |
| Acierto sobre las fotografías propias | **55.3 %** |
| CellSAM con preprocesamiento, mismas fotografías | 80.1 % |

El modelo aprendió a detectar colonias, con solo 27 imágenes de entrenamiento, y
aun así se degrada al cambiar de dominio. Combinado con la medición morfológica,
esto apunta a que la brecha está en las **condiciones de captura** y no en la
biología del organismo.

Dos defectos detectados en ese experimento y ya corregidos en el cuaderno: el
tope de detecciones por defecto de 300, que impedía contar enteras las placas con
más colonias, y una partición del ajuste que dejaba 29 colonias en entrenamiento
frente a 828 en validación.

---

## Fase 8. Datos sintéticos

**Motivo.** El fine-tuning de CellSAM falló con 16 placas y la hipótesis fue
falta de datos. Anotar miles de colonias a mano no es viable, así que se
construyó un generador (`scripts/generar_sinteticas.py`).

### Qué se tuvo en cuenta al construirlas

**Materia prima real, no dibujada.** Las colonias no se sintetizan por fórmula,
se **recortan de las fotografías propias** usando las coordenadas del conteo
manual, y se pegan sobre **fondos de placa reales** tomados de las placas casi
estériles del proyecto. Tanto la textura de la colonia como la del agar son
auténticas.

**Anotación exacta y gratuita.** Como el generador decide dónde pega cada
colonia, conoce su posición sin error. No hay ambigüedad de anotación, que es la
principal fuente de ruido en los conjuntos anotados a mano.

**Mezcla con borde suavizado.** Cada colonia se integra con una máscara circular
difuminada en lugar de pegarse como un cuadrado. Sin eso quedaría un contorno
recto que el modelo aprendería a reconocer como si fuera una característica de
las colonias.

**Ajuste de intensidad al destino.** Antes de pegar, se iguala el brillo del
anillo exterior del parche con el del fondo donde va a caer. Sin este paso una
colonia recortada de una zona clara conservaría ese brillo al pegarse en una
zona oscura, y se notaría de inmediato.

**Filtro de calidad de los parches.** Se exige que el centro del recorte sea más
oscuro que su borde, que es la definición de una colonia sobre agar, y se
descartan los parches con píxeles quemados o con demasiada varianza. Se añadió
tras observar medialunas blancas en la primera versión, procedentes del punto
caliente del centro de las fotografías. Descarta el 46 % de los parches
disponibles, de 796 a 429.

**Reparto por área, no por radio.** Las posiciones se sortean con la raíz
cuadrada del radio, de modo que la densidad sea uniforme sobre la superficie. Un
sorteo ingenuo concentraría las colonias en el centro.

**Solape permitido pero limitado.** Se admite que las colonias se toquen, porque
ocurre en las placas densas, pero no que se superpongan por completo.

**Densidad en escala logarítmica.** El número de colonias por placa se sortea
entre 5 y 320 en escala logarítmica, para que el conjunto tenga tanto placas
escasas como densas, igual que el material real.

**Variación de condiciones de captura** (opción `--variar-captura`). Pensada para
que un modelo entrenado con estas placas no dependa del montaje de un
laboratorio concreto. Se midió que entre las tres colecciones disponibles el
tono del agar va de 45 a 110, el gradiente de iluminación de 99 a 122 y el ruido
de 18 a 30, y la variación aplicada cubre ese rango. Incluye balance de blancos
por canal, gradiente de iluminación desde un punto arbitrario, viñeteo, cambio
de exposición, ruido de sensor, desenfoque leve y compresión JPEG agresiva.

**Resultados.**

| Conjunto | Placas | Colonias | Variación de captura |
|----------|--------|----------|---------------------|
| `datasets/sinteticas_yolo` | 600 | 46.591 | no |
| `datasets/sinteticas_variadas` | 800 | 60.211 | sí |

El segundo supera en volumen anotado al conjunto ADBC de la literatura, que
tiene 56.865 colonias.

### Lo que estas imágenes no capturan

Revisadas por la investigadora, el aspecto general no resulta convincente. Las
limitaciones identificadas, en orden de gravedad, son las siguientes.

**Las colonias se repiten.** Salen de un catálogo de 429 parches reutilizados
miles de veces, de modo que el mismo objeto aparece muchas veces en el conjunto.
Un modelo puede memorizar esas apariencias concretas en lugar de aprender el
concepto de colonia.

**Falta variación de tamaño.** En una placa real conviven colonias de edades
distintas, con un rango de tamaños amplio. Aquí todas parten del mismo radio de
recorte y solo se escalan entre 0.7 y 1.35, un rango demasiado estrecho.

**La distribución espacial es artificial.** Las posiciones se sortean de forma
independiente y uniforme, mientras que en una placa real las colonias se agrupan
por el modo de siembra, siguen el trazo del asa o se concentran donde cayó una
gota. Esa estructura espacial no se reproduce.

**No hay crecimiento ni interacción.** Las colonias reales crecen, se fusionan
cuando se tocan, generan colonias satélite y compiten por el medio. Aquí son
objetos rígidos pegados uno junto a otro.

**Falta coherencia de iluminación.** Una colonia real proyecta una sombra suave
y presenta un brillo especular consistentes con la dirección de la luz de la
escena. Los parches conservan la iluminación de su fotografía de origen, que no
coincide con la del fondo donde se pegan.

**Algunas variaciones de color son excesivas.** El rango de balance de blancos
llega a producir placas de tono rosado que no corresponden a ninguna fotografía
plausible de laboratorio.

**Consecuencia y siguiente paso.** Estas imágenes sirven para dar volumen a un
entrenamiento, pero no deben usarse nunca para evaluar. La evaluación tiene que
hacerse sobre fotografías reales, y en particular sobre el segundo lote, que no
participó en nada. Si el modelo entrenado con sintéticas no transfiere a las
reales, la causa más probable serán las limitaciones anteriores, y el orden
recomendado para atacarlas es ampliar el catálogo de colonias, ampliar el rango
de tamaños, introducir agrupamiento espacial y acotar la variación de color.

---

## Fase 9. Validación cruzada y exclusiones

**Exclusión de las series M1 y M2.** La investigadora determinó que en esas
placas la densidad del agar impidió el crecimiento colonial, de modo que
constituyen cultivos fallidos y no casos de prueba válidos. Son 7 placas con 29
colonias.

| Conjunto | MAE contable | Acierto |
|----------|--------------|---------|
| Las 15 placas | 4.08 | 80.1 % |
| **Sin M1 ni M2** | **2.83** | **83.8 %** |

La exclusión mejora los resultados de forma sustancial, porque esas placas
concentraban los falsos positivos, con 36 de error absoluto acumulado frente a 29
colonias reales. **Por eso la justificación debe presentarse de forma explícita y
biológica**, indicando cuántas placas se excluyen y por qué, nunca como un filtro
silencioso.

**El pipeline nuevo aplicado al conjunto de referencia**
(`results/colonias/placas_dobles/`). Las mejoras se desarrollaron sobre las
fotografías propias, así que este conjunto funciona como validación casi
independiente.

| | MAE | Sesgo |
|---|-----|-------|
| Pipeline histórico | **5.94** | −0.9 |
| Pipeline actual | **11.38** | **+8.50** |

**El pipeline nuevo empeora aquí, y de forma sistemática**, sobrecontando en 14
de las 16 placas.

**Primera hipótesis, descartada por medición.** Se supuso que la causa era la
corrección de iluminación, por estar diseñada para un gradiente de contraluz que
estas fotografías no tienen. Se ejecutó el conjunto completo sin ella
(`results/colonias/placas_dobles_sin_flat/`) y el resultado la descarta.

| Variante | MAE | Sesgo |
|----------|-----|-------|
| Pipeline histórico | **5.94** | −0.9 |
| Actual, con corrección | 11.38 | **+8.50** |
| Actual, sin corrección | 10.25 | **+8.50** |

El sesgo es idéntico en ambos casos, de modo que la corrección de iluminación no
interviene en el fenómeno.

**Segunda hipótesis, en comprobación: el umbral de detección.** El pipeline
histórico asignaba umbrales de 0.65 a 0.80 según la densidad estimada. El
selector actual, recalibrado para las fotografías con contraluz, asigna 0.40 a
todo lo que no sea muy denso. Estas placas tienen entre 13 y 68 colonias, así que
todas reciben ese umbral, casi la mitad de permisivo que el original. Un umbral
más bajo produce más detecciones, lo que explicaría un sesgo positivo constante.
Se comprueba ejecutando el conjunto con umbral fijo en 0.80.

Si se confirma, sería el **tercer caso del mismo patrón**, junto al estimador de
densidad y al filtro de color: parámetros calibrados en un montaje de captura que
no transfieren a otro.

---

## Estado actual

**Mejor configuración.** CellSAM base con recorte al 92 % del radio, corrección
de iluminación de campo plano, umbral adaptativo y filtros de área, solidez y
color.

| Conjunto | MAE contable | Acierto |
|----------|--------------|---------|
| Fotografías propias, sin cultivos fallidos | **2.83** | **83.8 %** |
| Fotografías propias, las 15 | 4.08 | 80.1 % |
| Conjunto de referencia, pipeline histórico | 5.94 | sin medir |

**Datos disponibles.**

| Conjunto | Placas | Colonias anotadas |
|----------|--------|-------------------|
| `images/placas` | 16 | 819, solo totales |
| `images/mis_fotos` | 15 | 856, con coordenadas |
| `images/mis_fotos_lote2` | 4 | prueba ciega, sin contar |
| `datasets/sinteticas_yolo` | 600 | 46.591 |

**Pendiente.**

- Confirmar que la corrección de iluminación es la causa del sobreconteo en el
  conjunto de referencia
- Reintentar el fine-tuning con las placas sintéticas, en GPU
- Descargar el dataset AGAR completo, que requiere registro, y repetir el
  entrenamiento de YOLO para que el resultado sea concluyente
- Contar a mano el segundo lote, para cerrar la prueba ciega
- Evaluar Cellpose y Omnipose con GPU
- Ampliar el conjunto anotado y medir la concordancia entre varios anotadores

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

## Fase 10. Tercer lote y el problema de la rotulación

Diez placas nuevas, la serie `RC73`, fotografiadas con mejor iluminación que los
lotes anteriores. La investigadora conserva el conteo manual sin comunicarlo, de
modo que **todo el desarrollo de esta fase se hizo a ciegas**. Esa restricción
resultó útil, porque obliga a justificar cada decisión por su fundamento y no
por el número que produce, que es justamente la crítica habitual al ajuste de
parámetros sobre el conjunto de prueba.

### El obstáculo: la rotulación con marcador

Estas placas están rotuladas a mano sobre el plástico, y el detector propone
regiones sobre los trazos igual que sobre las colonias. En las placas más
escritas llegó a proponer más trazos que colonias, hasta 28 frente a 14, de modo
que sin tratar la rotulación el conteo **se duplicaba**.

Costó cinco intentos, y los tres primeros fallaron por el mismo motivo.

| Intento | Criterio | Total | Qué falló |
|---------|----------|-------|-----------|
| 1 | Rango de tono fijo, 60 a 140 | 343 | El tono del marcador varía entre placas |
| 2 | Tono del agar medido, saturación ≥ 1.5 × la del agar | 260 | En agar saturado el umbral sube tanto que deja pasar la tinta |
| 3 | Saturación ≥ 0.9 × la del agar, mediana del tono | 192 | Descartaba colonias reales junto a la tinta |
| 4 | Mediana **y** percentil 90 del tono | 192 | Correcto, pero descarta la colonia que toca un trazo |
| 5 | **Eliminar la tinta antes de segmentar** | 206 | Ver abajo |

**Por qué los cuatro primeros comparten el mismo defecto.** Todos filtran
*después* de detectar, y eso obliga a decidir sobre regiones que montan a medias
sobre la escritura, donde la estadística de color es ambigua por construcción. Si
una colonia toca un trazo, el detector las une en una sola región y ya no hay
forma de separarlas: o se descartan ambas o se aceptan ambas.

**La solución.** Quitar la tinta antes de segmentar, con inpainting de Telea
(`scripts/quitar_rotulacion.py`). Es legítimo porque la rotulación está sobre el
plástico y no en el agar, así que es una oclusión del recipiente y no parte de la
muestra; lo que se reconstruye debajo es agar, que es liso y predecible. El
detector deja de proponer nada sobre ella, y la colonia vecina se detecta limpia.
El filtro por región se conserva como red de seguridad, y en 9 de 10 placas pasó
a descartar cero, lo que confirma que la eliminación previa hizo el trabajo.

**El efecto secundario, y su corrección.** La placa RC73-8 pasó de 6 a 15
colonias tras la eliminación, en dirección contraria a lo esperado. La inspección
visual de `RC73-8_count.png` mostró la causa: el inpainting dejaba **muescas
dentadas en el borde de la placa**, que el detector tomaba por colonias, unas
diez. Ocurre porque allí el trazo está pegado al límite del recorte y el
algoritmo no tiene vecindario válido del que copiar.

La corrección es no pretender reconstruir lo irreconstruible. La rotulación que
toca el borde **se excluye del área analizada** en lugar de rellenarse, con lo
que ni se reconstruye mal ni se cuenta. El coste es perder una franja estrecha
del borde donde, de todos modos, la escritura impide ver si hay colonias.

### Conceptos físicos aplicados al preprocesamiento

**Estimación del fondo por morfología** (`scripts/preproceso_fisico.py`). El
desenfoque gaussiano que se usaba hasta aquí se contamina con las propias
colonias, porque promedia todo lo que hay en la ventana. Una apertura en escala
de grises con un elemento mayor que la colonia más grande elimina los objetos
claros por construcción, así que estima el agar sin mezclarlo con lo que se
quiere medir.

El radio se eligió midiendo la rugosidad del fondo resultante, no a ojo:

| Radio | Rugosidad del fondo |
|-------|---------------------|
| 55 | 0,103 |
| 80 | 0,039 |
| **110** | **0,018** |
| 150 | 0,013 |

A partir de 110 px la mejora se aplana, de modo que se fija ahí: es el menor
radio que ya separa el fondo de las colonias.

**Densidad óptica por Beer-Lambert.** La luz que atraviesa una colonia cae de
forma exponencial con su biomasa, así que el logaritmo de la razón entre imagen y
fondo, `OD = -log10(I/I0)`, es proporcional a esa biomasa. Corrige la
iluminación y linealiza la respuesta en un solo paso, y da más contraste a las
colonias tenues que la imagen cruda.

### Resultado de la prueba ciega: las tres variantes convergen

Tras corregir la eliminación de la rotulación se contó el lote completo con tres
preprocesamientos de fundamento distinto, sin disponer del recuento manual.

| Placa | Imagen cruda | Densidad óptica | Mosaico |
|-------|-------------:|----------------:|--------:|
| RC73-1 | 21 | 21 | 19 |
| RC73-2 | 30 | 34 | 30 |
| RC73-3 | 24 | 25 | 24 |
| RC73-4 | 21 | 23 | 23 |
| RC73-5 | 25 | 25 | 25 |
| RC73-6 | 11 | 13 | |
| RC73-7 | 15 | 12 | |
| RC73-8 | 6 | 9 | |
| RC73-9 | 19 | 21 | |
| RC73-10 | 21 | 21 | 21 |
| **Total** | **193** | **204** | **192** |

La corrección del borde se confirma en RC73-8, que pasó de 15 a 6 colonias, y el
total bajó de 206 a 193 al desaparecer las muescas del inpainting.

**Lo relevante es que las tres convergen.** Son tres caminos distintos, la imagen
directa, la densidad óptica de Beer-Lambert sobre fondo morfológico, y el
troceado a mayor resolución, y el total se mueve entre 193 y 204, un 5 %. La
densidad óptica se separa de la imagen cruda en 1,1 colonias por placa de media,
en un rango de menos tres a más cuatro, y coincide exactamente en tres de diez.

Eso permite algo que normalmente exige el recuento manual: **una estimación de la
incertidumbre del método**. El conteo del lote está entre 193 y 204 colonias, con
una dispersión entre métodos del orden del 5 %. Si el recuento manual cae dentro
de esa banda, el procedimiento es consistente; si cae claramente fuera, existe un
sesgo sistemático compartido por las tres variantes, y eso sería a su vez un
resultado.

Lo que **no** se puede afirmar a ciegas es cuál de las tres es la correcta. La
conclusión defendible es más modesta y también más útil: la elección del
preprocesamiento no domina el resultado, que era justamente la duda.

### Consenso entre umbrales, sin conocer la respuesta

Sin conteo manual no hay forma de elegir un umbral mirando el resultado, y
elegirlo a ojo sería arbitrario. El consenso evita esa elección: se corre el
detector con cuatro umbrales, de 0,30 a 0,60, y se conserva cada colonia que
aparece en al menos la mitad de las corridas. Una detección estable entre
configuraciones es probablemente una colonia; una que solo aparece con el umbral
más permisivo es probablemente ruido.

Es el mismo principio de la votación entre modelos, aplicado a un solo modelo con
distintas sensibilidades, y **no requiere conocer la respuesta**, lo que lo hace
apropiado para un laboratorio que estrena el sistema.

---

## Fase 11. Qué separa un trazo de una colonia, y qué no

Al llevar el conteo a mayor resolución se vio que la eliminación de la
rotulación solo alcanza al **43 %** de los píxeles oscuros de la placa. La causa
es que el marcador de estas placas es negro, y el negro no tiene tono, de modo
que un criterio basado en el color no puede alcanzarlo. Se buscó entonces un
criterio que no dependiera del color.

### El descriptor de forma, y por qué parecía resolverlo

Un trazo es largo y estrecho, mientras que una colonia es redonda. Eso se resume
en una sola cifra independiente de la escala, el cociente entre el área del
componente y el área del mayor disco que cabe dentro de él:

> alargamiento = área / (π · semiancho²)

Vale 1 para un disco perfecto y crece con la relación entre largo y ancho. Medido
sobre el tercer lote, separa de forma aparentemente perfecta:

| | Alargamiento |
|---|---|
| Colonias | 1,04 a 1,96 |
| Trazos | 3,25 a 16,7 |

Hay un hueco vacío entre 2 y 3, lo que invitaba a fijar el corte en 3,0.

### El control negativo lo desmonta

Antes de adoptarlo se aplicó a las **placas dobles, que son densas y no tienen
escritura alguna**. Si el descriptor midiera lo que se pretende, allí no debería
marcar casi nada.

| Conjunto | Componentes oscuros | Con alargamiento > 3 |
|----------|--------------------:|---------------------:|
| Placas dobles, sin escritura | 44 | **31 (70,5 %)** |
| Primer lote, contraluz | 406 | 36 (8,9 %) |
| Tercer lote, con escritura | 271 | 83 (30,6 %) |

En las placas densas, las colonias se tocan y forman cadenas, y una cadena es
tan alargada como un trazo. La regla habría **borrado la mayor parte de las
colonias justo en las placas más pobladas**, que son las que más importan. El
descriptor no mide escritura, mide alargamiento, y ambas cosas coinciden solo
cuando la placa está poco poblada.

### Tres intentos de rescatarlo, los tres fallidos

Se probaron tres propiedades que deberían distinguir un trazo de ancho constante
de una cadena de colonias con bultos, comparando siempre contra el control
negativo:

| Descriptor | Densas sin escritura | Lote 3 con escritura |
|------------|---------------------:|---------------------:|
| Número de núcleos | 2,00 | 2,00 |
| Área explicada por discos inscritos | 2,69 | 3,61 |
| Oscuridad relativa al agar | 0,76 | 0,71 |

Las distribuciones se solapan casi por completo. La posición tampoco sirve: el
71 % de los componentes alargados de las placas sin escritura también llega al
borde, porque allí el propio anillo del menisco es un componente oscuro y
alargado.

### Lo que queda establecido

**El color es la única propiedad que separa el marcador de la biomasa.** No es
un accidente sino algo esperable, porque el pigmento del rotulador y el de la
colonia son sustancias distintas, mientras que su forma y su brillo pueden
coincidir.

De ahí se sigue una **limitación que hay que declarar**: el sistema trata bien el
marcador de color y no puede tratar el marcador negro escrito sobre la zona de
cultivo. La mitigación no es algorítmica sino de protocolo, y no cuesta nada:
rotular en el reverso o en el anillo exterior, nunca cruzando el área sembrada.
Es además buena práctica por sí misma, porque la escritura sobre el cultivo
también estorba al conteo manual.

Un resultado lateral sí es aprovechable. Las colonias redondas **nunca** superan
el 0,90 del radio en las placas densas, y solo el 18 % lo hace en el tercer lote.
Eso respalda por una vía independiente el recorte al 92 % del radio que se había
fijado con la curva de recuperación.

---

## Fase 12. Revisión de literatura, septiembre de 2026

Se revisó el estado del arte justo después de implementar la inferencia por
mosaico, y el resultado obliga a reubicar la aportación.

### La idea del mosaico ya está publicada, y funciona

Un artículo de 2026 en *Scientific Reports* hace exactamente lo mismo, con el
mismo razonamiento: los detectores fallan porque la imagen de alta resolución se
reduce a 640 × 640 y las colonias pequeñas desaparecen. Parten la imagen en
baldosas de 640 × 640 con un 20 % de solape, que es casi la configuración que se
eligió aquí de forma independiente.

| Inferencia | mAP@0.5 |
|------------|---------|
| Redimensionando la placa entera | 44,9 a 66,3 % |
| **Por baldosas** | **95,4 a 96,9 %** |

El salto es enorme y **confirma el razonamiento físico** de que el problema está
en la reducción de escala, no en el detector. También significa que el mosaico
**no puede presentarse como aportación propia**, sino como una técnica conocida
que se adopta y se cita.

Tres detalles agravan la coincidencia: usan fotografías de teléfono, de tres
modelos distintos, y explícitamente «sin iluminación normalizada ni protocolo
estricto de posicionamiento». Es el mismo escenario de laboratorio sin
presupuesto que motiva este trabajo.

### El uso de modelos fundacionales también

*Colony Grounded SAM2* combina Grounding DINO con SAM 2 para detectar y
segmentar colonias prácticamente sin entrenamiento, con 93,1 % de precisión
media sobre el conjunto ADBC. No trata actinomicetos ni organismos filamentosos.

### Dónde queda entonces la aportación

Lo valioso está en lo que esos trabajos declaran como pendiente, porque coincide
punto por punto con lo que aquí ya está medido:

| Lo que ellos dejan pendiente | Lo que aquí hay |
|------------------------------|-----------------|
| «validación externa en distintos laboratorios y dispositivos» | El hallazgo central: tres parámetros calibrados en un montaje fallan en otro, y la auto-calibración que lo corrige |
| «evaluación sobre un único conjunto público» | Tres lotes propios, de captura independiente, con conteo manual |
| «detección por caja en lugar de segmentación de instancias» | CellSAM da segmentación de instancias |
| «las colonias densamente agrupadas del centro se pierden» | La separación por cuencas de `separar_pegadas.py` |
| Ninguno trata actinomicetos | El objeto de estudio de este trabajo |

La conclusión práctica es que **el método no es la aportación, la
transferibilidad sí**. Nadie ha demostrado que un contador entrenado o ajustado
en un laboratorio siga funcionando en otro con otro teléfono y otra luz, y aquí
está medido que no, con las tres formas concretas en que falla.

### Conjunto de datos que conviene incorporar

ADBC, publicado en *Scientific Data* en 2023, tiene 369 placas de 24 especies
con 56.865 colonias anotadas, tomadas con tres teléfonos distintos y sin
iluminación normalizada. Está libre en Figshare. Sirve para dos cosas: validar
el pipeline sobre capturas de otro laboratorio, que es justo lo que se quiere
demostrar, y como precedente de que un artículo de conjunto de datos con este
material entra en *Scientific Data*.

**Referencias.**

- Overcoming resolution constraints in automated colony counting via a
  high-performance deep learning framework using SAHI. *Scientific Reports*,
  2026. https://www.nature.com/articles/s41598-026-55724-1
- Colony Grounded SAM2: Zero-shot detection and segmentation of bacterial
  colonies using foundation models. arXiv:2603.13393
- Annotated dataset for deep-learning-based bacterial colony detection.
  *Scientific Data*, 2023. https://www.nature.com/articles/s41597-023-02404-8
- Enhancing Colony Detection of Microorganisms in Agar Dishes Using SAM-Based
  Synthetic Data Augmentation in Low-Data Scenarios. *Applied Sciences*, 2025.
  https://doi.org/10.3390/app15031260
- AGAR, a microbial colony dataset for deep learning detection. arXiv:2108.01234

---

## Fase 13. Validación externa con ADBC

Se descargó el conjunto ADBC (`scripts/bajar_adbc.py`), 369 placas de 24
especies con 56.865 colonias anotadas, tomadas con tres teléfonos distintos y
sin iluminación normalizada, con licencia CC BY 4.0. Es la validación externa
que los trabajos recientes declaran como pendiente.

### Qué tiene dentro

| Densidad, colonias por placa | Placas | Proporción |
|------------------------------|-------:|-----------:|
| 1 a 10 | 39 | 10,6 % |
| 11 a 30 | 52 | 14,1 % |
| 31 a 60 | 51 | 13,8 % |
| 61 a 150 | 100 | 27,1 % |
| 151 a 250 | 52 | 14,1 % |
| Más de 250, incontables | 75 | 20,3 % |

La mediana es de 108 colonias por placa y el máximo 747, de modo que **es un
conjunto bastante más denso que el de este trabajo**, donde las placas rondan
las 20 colonias. Eso lo convierte en una prueba exigente y no en una
confirmación cómoda.

Un dato que conviene citar en el capítulo: **hay 344 tamaños de imagen
distintos en 369 placas**. Casi cada fotografía tiene su propia resolución, que
es justamente el desorden de captura que este trabajo sostiene que hay que
tolerar en vez de corregir con protocolo.

### Predicción registrada antes de ejecutar

Se deja escrita por adelantado para que el resultado pueda contradecirla. Al
reducir la placa a los 1024 píxeles que CellSAM usa internamente, el diámetro de
la colonia mediana queda así:

| Densidad | Diámetro tras reducir |
|----------|----------------------:|
| 1 a 10 | 38,8 px |
| 11 a 30 | 33,5 px |
| 31 a 60 | 36,1 px |
| 61 a 150 | 30,0 px |
| 151 a 250 | 26,0 px |
| Más de 250 | **19,3 px** |

Los detectores pierden fiabilidad por debajo de unos 15 a 20 px, y el 57 % de
las placas incontables caen bajo ese umbral.

> **Predicción.** La inferencia por mosaico debe mejorar el conteo en las placas
> densas y resultar indiferente, o levemente peor por fragmentación, en las
> ralas. Si el resultado sale al revés, el razonamiento físico sobre la
> reducción de escala es incorrecto y hay que abandonarlo.

### Dos errores encontrados al salir del laboratorio propio

Ninguno de los dos se habría visto nunca con las fotografías propias, y los dos
aparecieron al primer contacto con placas ajenas. Son la mejor ilustración
posible de lo que este trabajo sostiene.

**Primero, el tono tratado como número y no como ángulo.** El matiz es un
ángulo, y en la escala de OpenCV, que va de 0 a 179, el valor 179 está pegado al
0. Los agares de sangre de ADBC tienen su tono en 177 a 179, justo en esa
costura. Una colonia de tono 2 dista 4 grados del agar de tono 178, pero la
resta directa da 176, de modo que el filtro la tomaba por rotulación y **la
borraba en silencio**. En las fotografías propias no se manifestó nunca porque
su agar está en 41 a 56, lejos de la costura.

Corregirlo tuvo más recorrido del esperado, porque las dos salidas obvias
fallan, y ambas se comprobaron antes de adoptar nada:

| Estimador del tono del agar | Qué rompe |
|-----------------------------|-----------|
| Mediana | Da 90, un verde inexistente, si el agar cruza la costura |
| Moda circular | Los píxeles grises tienen tono 0 por convenio y se amontonan: una placa propia con zona oscura daba 0 frente a 46 |
| Moda ponderada por saturación | Una región pequeña e intensa gana a un agar grande y pálido: una placa doble daba 21, un naranja, frente a 87 |
| **Descartar donde el tono no está definido y contar por área** | Nada. Es el que quedó |

Comprobado sobre veinte placas de cuatro procedencias: coincide con la mediana
dentro de un grado en todas las placas propias, y dentro de un mismo montaje es
**más consistente** que ella, con 8 grados de dispersión frente a 23, que es lo
que debe ocurrir si las placas se fotografiaron igual.

**Segundo, el área mínima fijada en píxeles.** El pipeline exigía 300 px², que
traducido a unidades reales significa exigir que una colonia mida 1,35 mm.
Medido sobre las anotaciones de ADBC:

| Tramo de densidad | Colonias anotadas | Bajo el filtro de 300 px² |
|-------------------|------------------:|--------------------------:|
| 1 a 10 | 185 | 17,3 % |
| 11 a 30 | 1.014 | 2,1 % |
| 31 a 60 | 2.252 | 5,0 % |
| 61 a 150 | 11.160 | 8,2 % |
| 151 a 250 | 10.532 | 22,9 % |
| Más de 250 | 31.722 | **40,6 %** |

En total se habría descartado el 28,8 % de las colonias anotadas, de modo que la
validación externa habría medido el filtro y no el detector. Es el **cuarto caso
del mismo patrón**, junto al estimador de densidad, el filtro de color y el
umbral de detección: una constante fijada para un montaje concreto.

La corrección es la misma que en los otros tres, poner la regla en unidades que
signifiquen algo fuera de ese montaje. Como el recorte se lleva siempre a un
lado fijo y abarca una fracción conocida de una placa normalizada de 90 mm, la
escala se deduce sin calibrar nada. El mínimo se declara ahora en milímetros.

El evaluador guarda además el área de cada detección y no solo el recuento, de
modo que ese umbral se puede mover después sin volver a ejecutar el modelo, y la
sensibilidad a él se informa junto al resultado en lugar de quedar escondida.

### Limitación de cómputo

No hay GPU disponible, y CellSAM sobre procesador tarda del orden de minutos por
placa, de modo que las 369 no son viables. Se trabaja sobre una **submuestra
estratificada** con el mismo número de placas por tramo de densidad, elegida con
semilla fija para que sea reproducible. Estratificar y no muestrear al azar
importa aquí, porque el efecto que se quiere medir depende justamente de la
densidad.

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
| `images/mis_fotos_lote3` | 10 | prueba ciega, conteo en poder de la investigadora |
| `datasets/sinteticas_yolo` | 600 | 45.991 |
| `datasets/sinteticas_variadas` | 800 | 59.411 |
| **`datasets/sinteticas_v3`** | **1000** | **71.522** |

**Pendiente.**

- Confirmar que la corrección de iluminación es la causa del sobreconteo en el
  conjunto de referencia
- Reintentar el fine-tuning con las placas sintéticas, en GPU
- Descargar el dataset AGAR completo, que requiere registro, y repetir el
  entrenamiento de YOLO para que el resultado sea concluyente
- Contar a mano el segundo lote, para cerrar la prueba ciega
- Comparar el conteo del tercer lote con el conteo manual que conserva la
  investigadora, que es la unica forma de cerrar esa prueba
- Medir si la inferencia por mosaico recupera colonias pequenas, y a que coste
  en falsos positivos
- Evaluar Cellpose y Omnipose con GPU
- Ampliar el conjunto anotado y medir la concordancia entre varios anotadores

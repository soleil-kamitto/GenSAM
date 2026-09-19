# CAPÍTULO 1

---

## 1.1 Introducción

Los actinomicetos son uno de los grupos bacterianos con mayor importancia biotecnológica a nivel mundial, ya que estas bacterias Gram-positivas filamentosas, presentes sobre todo en los suelos, se encuentran entre las principales fuentes de antibióticos de origen natural. Cerca del 45 % de los metabolitos microbianos bioactivos conocidos proviene de actinomicetos (Bérdy, 2005), y se estima que el género *Streptomyces* por sí solo produce alrededor del 75 % de los antibióticos de utilidad clínica y comercial (Watve et al., 2001), entre ellos la estreptomicina, la eritromicina, la tetraciclina y la vancomicina. La resistencia antimicrobiana fue responsable directa de cerca de 1,27 millones de muertes en el mundo durante 2019 (Murray et al., 2022), y podría llegar a 10 millones de muertes anuales hacia 2050 si las tendencias actuales se mantienen (O'Neill, 2016), de modo que la búsqueda sistemática de nuevas cepas de actinomicetos con actividad biológica es hoy una prioridad de investigación. Esta búsqueda depende del cultivo en placas de Petri y del conteo de las colonias características, porque a mayor número de muestras analizadas, mayor es la probabilidad de encontrar cepas que produzcan compuestos bioactivos nuevos (Tiwari & Gupta, 2012).

El procedimiento estándar en los laboratorios de microbiología ambiental consiste en sembrar diluciones de muestras de suelo en agar selectivo, incubar las placas durante 7 a 14 días y contar de forma manual las colonias características de actinomicetos, que se reconocen por su superficie polvosa o dura, su color variable y su aspecto esporulado. En los programas de prospección de gran escala, cada muestra de suelo se siembra en varias diluciones y réplicas sobre medios selectivos (Adegboye & Babalola, 2012), de modo que una sola campaña de muestreo puede generar decenas o cientos de placas, y cada una exige una inspección visual cuidadosa cuya duración crece con la densidad de colonias. Este cuello de botella limita de manera importante la capacidad de cribado de las instituciones de investigación.

En los últimos años, la combinación de modelos de segmentación profunda con grandes colecciones de imágenes biológicas ha producido herramientas de propósito general para el análisis de imágenes de células. Cutler et al. (2022) desarrollaron Omnipose, una red neuronal basada en U-net que supera a los métodos clásicos de umbralización en la segmentación precisa de células bacterianas de forma arbitraria, y más recientemente Marks et al. (2025) publicaron CellSAM, un modelo de fundación (*foundation model*) construido sobre la arquitectura SAM (*Segment Anything Model*) y el detector AnchorDETR, entrenado con un corpus diverso de imágenes de células de múltiples modalidades de microscopía. CellSAM muestra capacidad de generalización hacia tipos de células que no estaban en su entrenamiento original (Marks et al., 2025), y por esa razón es un buen candidato para adaptarlo a dominios nuevos si la imagen de entrada se prepara de forma adecuada.

El presente trabajo propone un sistema automático de conteo de colonias de actinomicetos en imágenes macroscópicas de placas de Petri tomadas con equipos comunes, como teléfonos inteligentes o cámaras de bajo costo, y el sistema se basa en CellSAM con una etapa de preprocesamiento de ecualización adaptativa del histograma (CLAHE) y un módulo de selección adaptativa del umbral de detección. El sistema se evaluó sobre dos conjuntos de imágenes, que son un dataset de referencia con conteo manual de colonias y un conjunto de 15 placas sembradas y fotografiadas por la propia investigadora en condiciones de captura distintas, sobre un transiluminador. Como el diseño trabaja con imágenes de celular en lugar de microscopía especializada, se elimina la barrera del equipamiento y el sistema puede ofrecerse como un servicio web accesible desde el campo, donde el investigador sube sus fotografías de placa y recibe el conteo automático junto con la visualización de la segmentación por instancia.

---

## 1.2 Descripción del Problema

El conteo manual de colonias de actinomicetos en placas de Petri es hoy el procedimiento estándar en la mayoría de los laboratorios de microbiología ambiental y farmacéutica. En este procedimiento, el operador inspecciona cada placa bajo iluminación directa, identifica las colonias características por su forma y su color, y las cuenta una por una. Este método presenta cuatro limitaciones principales.

**Tiempo operativo elevado.** El tiempo de inspección de cada placa crece con la densidad de colonias y depende de la experiencia del operador, y como los protocolos de aislamiento siembran cada muestra en varias diluciones y réplicas sobre medios selectivos (Adegboye & Babalola, 2012), el conteo manual acumulado de una campaña de prospección termina ocupando una fracción desproporcionada del tiempo total de la investigación.

**Variabilidad entre observadores y tendencia al error.** Aunque el conteo manual sigue siendo el estándar de referencia, es lento y produce resultados distintos entre observadores, porque cada uno aplica su propio criterio para incluir las colonias marginales, superpuestas o difíciles de ver (Putman et al., 2005; Clarke et al., 2010). Además, en comparaciones directas contra un estándar de referencia, el conteo manual de rutina mostró desviaciones sistemáticas de subconteo, que fueron más marcadas en las placas de alta densidad (Brugger et al., 2012).

**Fatiga visual y error acumulado.** El conteo repetitivo es una tarea monótona y propensa al error (Clarke et al., 2010), de modo que las equivocaciones tienden a aumentar con el número de placas procesadas en una sesión y los resultados de un mismo estudio se vuelven menos comparables entre sí.

**Escalabilidad limitada.** Como el procedimiento depende de un operador humano, existe un techo operativo que hace inviable el análisis de grandes colecciones de muestras en plazos razonables, y ese techo limita el alcance de los programas de prospección de nuevos antibióticos (Tiwari & Gupta, 2012).

El problema se agrava por la forma propia de las colonias de actinomicetos, ya que presentan bordes difusos, textura granular o polvosa, colores variados (blanco, gris, naranja, rojo o marrón) y tendencia a superponerse en las placas de alta densidad. Las herramientas de análisis de imagen disponibles, como los plugins de conteo de colonias en ImageJ/Fiji o los contadores automáticos comerciales basados en umbralización clásica, fueron diseñadas y validadas para colonias compactas de bordes bien definidos sobre medios de contraste uniforme, y sus propios autores reconocen que las colonias de apariencia muy distinta o de bordes difusos exigen adaptar el algoritmo (Clarke et al., 2010; Brugger et al., 2012), que es justo el caso de los actinomicetos.

Por otro lado, los modelos de segmentación de células de última generación, como Omnipose (Cutler et al., 2022) y CellSAM (Marks et al., 2025), fueron entrenados y validados sobre todo con imágenes de microscopía, cuyas condiciones de contraste, escala y fondo son muy distintas a las de una placa macroscópica fotografiada con celular. En los experimentos de este trabajo, la aplicación directa de CellSAM sobre imágenes de placa sin preprocesamiento produjo cero detecciones, porque el modelo no distingue las colonias del fondo del agar sin una normalización de contraste previa. Este trabajo demuestra que ese problema se resuelve con CLAHE, y que con la configuración apropiada el modelo alcanza un error absoluto medio (MAE) de 5.94 colonias por placa, medido sobre las 16 placas del conjunto de referencia con conteo manual.

---

## 1.3 Justificación

**Urgencia del descubrimiento de nuevos antibióticos.** La resistencia antimicrobiana es una de las mayores amenazas para la salud pública de este siglo, ya que fue responsable directa de 1,27 millones de muertes en 2019 (Murray et al., 2022) y podría llegar a 10 millones de muertes anuales hacia 2050 si las tendencias actuales continúan (O'Neill, 2016). Los actinomicetos, y en particular el género *Streptomyces*, han sido la fuente de la mayor parte de los antibióticos de origen natural que hoy se usan en la clínica (Watve et al., 2001; Bérdy, 2005). Sin embargo, el descubrimiento de compuestos bioactivos nuevos se ha vuelto más lento, y encontrar cepas productoras exige aislar y cribar volúmenes crecientes de muestras (Tiwari & Gupta, 2012), de modo que el conteo manual de colonias se convierte en un cuello de botella. Un sistema que automatice este paso libera una parte importante del tiempo del investigador y amplía el espacio de búsqueda de cepas nuevas.

**Accesibilidad tecnológica para laboratorios de recursos limitados.** Las soluciones basadas en microscopía de contraste de fase requieren equipos especializados de alto costo, y la espectrometría MALDI-TOF, aunque se ha consolidado como un método rápido de identificación bacteriana (Szabó et al., 2022), exige una inversión inicial del orden de cientos de miles de dólares. A diferencia de ambas, el presente sistema trabaja con fotografías macroscópicas de placas de Petri tomadas con el teléfono del investigador, así que se elimina la barrera del equipamiento y el despliegue se vuelve viable en laboratorios universitarios, institutos de investigación regionales y programas de monitoreo ambiental en campo, contextos que son especialmente relevantes en el Perú y en América Latina (Acuña, 2023).

**Viabilidad técnica demostrada.** En este trabajo, CellSAM muestra capacidad de generalización hacia imágenes macroscópicas de actinomicetos con una modificación que no toca el modelo, porque solo se añade una etapa de preprocesamiento CLAHE que corrige el bajo contraste entre las colonias y el agar. Cuando esta etapa se complementa con un módulo de selección adaptativa del umbral de detección, que estima la densidad de colonias con visión clásica en milisegundos y asigna de forma automática el *bbox\_threshold* apropiado, el sistema alcanza un MAE de **5.94 colonias/placa** sobre el conjunto de referencia con conteo manual, formado por 8 imágenes con dos placas cada una (16 placas), y esa cifra representa una reducción del 53 % frente a la configuración base (MAE = 12.75). Estos resultados demuestran la viabilidad del enfoque sin necesidad de entrenar el modelo desde cero.

**Validación preliminar sobre imágenes propias.** Como primera prueba del sistema sobre material generado por la propia investigadora, se procesaron 15 fotografías de placas sembradas en el laboratorio con muestras de actinomicetos (series M1, M2, MC73 y RC73, con réplicas y, en algunos casos, diluciones), y las imágenes se capturaron con cámara de teléfono sobre un transiluminador, con una placa por imagen. Estas condiciones se diferencian del conjunto de referencia en la iluminación a contraluz, en la resolución de captura y en la rotulación con marcador sobre el borde de la placa. La adaptación del pipeline incorporó una corrección de iluminación de campo plano (*flat-field*), que resultó ser el hallazgo técnico más importante para este escenario, porque en las placas más afectadas por el contraluz bajar el umbral de detección no recuperaba ninguna detección, mientras que con la corrección aplicada el sistema sí recupera la estructura de la placa, y una placa de control conservó su conteo correcto de 13 colonias en todas las variantes del experimento. Contra el conteo manual de referencia construido por la propia investigadora, el sistema alcanzó un MAE de 4.08 colonias/placa sobre las 13 placas dentro del rango contable estándar (hasta 250 colonias), una cifra incluso mejor que el MAE de 5.94 del conjunto de referencia original, lo que indica que el enfoque se transfiere a condiciones de captura nuevas. Sobre el total de 856 colonias contadas a mano, el acierto agregado del sistema es del 80.1 %. Las dos placas restantes superan el rango contable convencional (TNTC) y presentan subconteo por fusión de colonias vecinas. La validación también dejó limitaciones documentadas, entre ellas la necesidad de recalibrar el selector adaptativo de umbral para estas condiciones, un subconteo persistente en las colonias pequeñas y pálidas, y falsos positivos sobre un crecimiento que la investigadora consideró no válido en una placa de la serie M2.

**Código abierto y escalabilidad.** El sistema está implementado por completo en Python con bibliotecas de código abierto (PyTorch, OpenCV, scikit-image), de modo que la comunidad científica puede auditarlo, reproducirlo y extenderlo. Además, el diseño como servicio web facilita su integración en los flujos de trabajo existentes sin instalar dependencias complejas, porque el investigador sube sus fotografías desde el celular y recibe en segundos el conteo automático con la imagen segmentada.

---

## 1.4 Estado del Arte

Los actinomicetos acumulan un volumen creciente de investigación científica, ya que en la base Scopus aparecen actualmente cerca de 45,000 publicaciones con el término *Actinomycetes* y más de 120,000 con *Streptomyces* (cifras aproximadas, que dependen de la sintaxis de búsqueda empleada), con aumentos sostenidos desde la década de 1990 impulsados por la búsqueda de compuestos bioactivos nuevos. Sin embargo, dentro de ese volumen, los trabajos dedicados de forma específica a la **automatización del conteo de colonias mediante visión artificial** siguen siendo escasos, como se ilustra en las Figuras 1.1 y 1.2.

> **[Figura 1.1, PENDIENTE]** Número de publicaciones por año en Scopus con keyword: *Actinomycetes* (1990–2024).
>
> **[Figura 1.2, PENDIENTE]** Número de publicaciones por año en Scopus con keywords: *Actinomycetes* AND *automated counting* (1990–2024).

Para poner en contexto el panorama tecnológico actual, la **Tabla 1.1** presenta los métodos convencionales de conteo de colonias que se emplean en los laboratorios de microbiología, y la **Tabla 1.2** resume las soluciones basadas en visión artificial y aprendizaje profundo que han aparecido en los últimos cinco años.

**Tabla 1.1. Métodos convencionales para el conteo de colonias en placas de Petri.**

| N° | Tecnología | Descripción | Costo estimado | Referencia |
|----|------------|-------------|----------------|------------|
| 1 | Conteo manual (visual) | Cuantificación visual por operador bajo iluminación directa o lupa. Sujeto a fatiga, lentitud y variabilidad entre observadores. | $15–30/hr (labor del operador) | Putman et al. (2005); Clarke et al. (2010) |
| 2 | Contadores automáticos comerciales (p.ej. aCOLyte, ProtoCOL) | Imagen de placa + software de umbralización clásica. Limitado a colonias de forma regular en medios claros. | $3,000–$8,000 (hardware + licencia) | Clarke et al. (2010); Brugger et al. (2012) |
| 3 | MALDI-TOF Spectrometry | Estándar clínico de identificación por perfil proteico. No realiza conteo de colonias, porque se usa en la identificación de especie posterior al aislamiento. | $150,000–$250,000 (adquisición inicial) | Szabó et al. (2022) |

**Tabla 1.2. Soluciones basadas en visión artificial y aprendizaje profundo para segmentación y conteo celular.**

| N° | Tecnología | Descripción | Costo aproximado | Referencia |
|----|------------|-------------|-----------------|------------|
| 1 | Omnipose (Deep Learning) | Red U-net con campo de gradiente de función de distancia. Alta precisión en células bacterianas de forma arbitraria en microscopía, y supera a los métodos clásicos en cultivos mixtos y formas alargadas o ramificadas. | $0 software + GPU | Cutler et al. (2022) |
| 2 | CellSAM (Foundation Model) | Modelo de fundación basado en SAM + AnchorDETR, entrenado con un corpus diverso de imágenes celulares de múltiples modalidades. Capacidad de generalización hacia tipos celulares no vistos en el entrenamiento. | $0 software + CPU/GPU | Marks et al. (2025) |
| 3 | ResNet-50 / Vision Transformer (ViT) | Clasificación automática de cepas resistentes a antibióticos mediante cambios de forma cuantificables en microscopía de luz, sin requerir pruebas bioquímicas adicionales. | No estimado | Ikebe et al. (2024) |
| 4 | **CellSAM + preprocesamiento adaptativo** *(presente trabajo)* | CellSAM adaptado a imágenes macroscópicas de placas de Petri mediante CLAHE, corrección de iluminación *flat-field* y selección adaptativa del *bbox\_threshold* según la densidad estimada de colonias. Operable con imágenes de celular. **MAE = 5.94 colonias/placa** en el conjunto de referencia con conteo manual (16 placas) y **MAE = 4.08** sobre las placas propias capturadas sobre transiluminador dentro del rango contable (13 placas). Código abierto. | $0 | Presente trabajo |

Como se observa en la Tabla 1.2, todavía existe un vacío en soluciones que combinen (i) segmentación profunda de alta precisión, (ii) operación sobre imágenes macroscópicas tomadas con equipos de bajo costo, (iii) adaptación a la forma particular de las colonias de actinomicetos y (iv) despliegue accesible como servicio web para investigadores en campo, y el presente trabajo se ubica precisamente en ese espacio.

Hallström et al. (2023; 2025) demostraron que la clasificación automática de especies bacterianas con aprendizaje profundo sobre imágenes de microscopía es viable con precisiones superiores al 93 %, aunque sus enfoques requieren chips microfluídicos y microscopios de contraste de fase, por lo que su uso queda limitado a laboratorios muy equipados. En el contexto latinoamericano, Acuña (2023) demostró que es viable construir sistemas de visión artificial con hardware impreso en 3D y control web para el conteo automático de microorganismos acuáticos, y ese antecedente metodológico se aplica de forma directa al presente trabajo.

### 1.4.1 Resultados preliminares sobre placas propias

Para cerrar este panorama conviene describir el estado actual del sistema desarrollado. Además del conjunto de referencia con conteo manual, el pipeline se aplicó a 15 fotografías de placas sembradas por la propia investigadora, y esa aplicación es la primera prueba sobre imágenes generadas por completo dentro del proyecto y con un montaje de captura distinto. Las placas corresponden a las series M1, M2, MC73 y RC73, con réplicas identificadas por letra y, en algunos casos, con la dilución indicada en el nombre, y la captura se realizó con cámara de teléfono a una resolución de hasta 4096 px, con la placa colocada sobre un transiluminador que la ilumina a contraluz, una placa por imagen y la rotulación escrita con marcador azul sobre el borde.

Estas condiciones traen tres dificultades que no estaban en el conjunto de referencia. La primera es un gradiente de iluminación pronunciado, propio del contraluz, que oscurece la periferia de la placa respecto del centro; la segunda es la rotulación con marcador dentro del campo de la imagen, que el segmentador tiende a proponer como si fuera una colonia; y la tercera es la banda de artefactos del borde de la placa, donde se juntan el menisco del agar y la condensación. El pipeline adaptado responde a las tres dificultades. Primero detecta la placa mediante la transformada de Hough y recorta al 92 % del radio, un valor que se fijó con la curva de recuperación descrita más abajo, y luego reescala el recorte a 1200 px, que es la escala con la que se calibró el pipeline original. Después aplica la corrección de iluminación de campo plano, que consiste en dividir la imagen por un fondo estimado con desenfoque gaussiano de núcleo igual al 15 % del lado, seguida de un recorte por los percentiles 1 y 99 y de CLAHE con *clipLimit* 2.0. Por último ejecuta CellSAM con `normalize=True`, `postprocess=True` y `bbox_threshold` 0.40, y descarta regiones por área (300 a 50000 px²), por solidez (mayor o igual a 0,50) y por color en el espacio HSV, y este último filtro rechaza la tinta del rotulador, que se midió en el rango de matiz 71 a 105 con saturación alta frente al rango 39 a 44 de las colonias.

La corrección de iluminación resultó ser el hallazgo decisivo para este escenario, y el experimento factorial que la sustenta se resume en la **Tabla 1.3**. En la placa M2-A-3, la más afectada por el contraluz, el sistema no proponía ninguna región sin corrección, y tampoco lo hacía al bajar el umbral, de modo que el problema no era la sensibilidad del detector sino el contraste de la imagen de entrada; con la corrección aplicada, el sistema pasó a proponer 15 y 29 regiones según el umbral. Sin embargo, el conteo manual posterior obligó a revisar este caso, como se detalla más abajo, porque la investigadora consideró no válido el crecimiento de esa placa y esas detecciones resultaron ser falsos positivos. Por lo tanto, la evidencia de la mejora se apoya en MC73-C, donde el conteo subió de 1 a 4 colonias frente a un conteo manual de 5, y en la placa de control MC73-A, que mantuvo su conteo correcto de 13 colonias en las cuatro variantes, lo que confirma que la corrección no daña los casos que ya funcionaban.

**Tabla 1.3. Efecto de la corrección de iluminación *flat-field* y del umbral de detección sobre tres placas propias.**

| Placa | Sin corrección, thr 0.65 | Sin corrección, thr 0.40 | Con corrección, thr 0.65 | Con corrección, thr 0.40 |
|-------|--------------------------|--------------------------|--------------------------|--------------------------|
| M2-A-3 (colonias pálidas en cadena) | 0 | 0 | 15 | **29** |
| MC73-C (fallo parcial) | 1 | 3 | 4 | **4** |
| MC73-A (control de no regresión) | 13 | 13 | 13 | **13** |

El filtro de color aporta una mejora independiente y comprobable, porque sobre una placa estéril la rotulación con marcador generaba 8 detecciones falsas antes de incorporarlo, y en el conjunto completo el filtro descartó entre 0 y 9 regiones por placa. Su contribución se midió desactivándolo sobre las 15 placas: sin él el MAE en rango contable sube de 4.00 a 7.31 colonias por placa y el acierto agregado baja del 80.3 % al 75.7 %, de modo que el filtro aporta algo más de cuatro puntos y medio de acierto por sí solo. La **Tabla 1.4** presenta los conteos del sistema frente al conteo manual de referencia de las 15 placas.

**Tabla 1.4. Conteo automático frente a conteo manual de referencia en las 15 placas propias.**

| Placa | Manual | Sistema | Error |
|-------|-------:|--------:|------:|
| M1-B | 2 | 2 | 0 |
| M1-C | 1 | 3 | +2 |
| M1-D | 21 | 8 | −13 |
| M2-A-3 | 0 ᵃ | 11 | +11 |
| M2-A | 3 | 9 | +6 |
| M2-B | 0 | 0 | 0 |
| M2-C | 2 | 6 | +4 |
| MC73-A | 13 | 16 | +3 |
| MC73-B | 18 | 15 | −3 |
| MC73-C | 5 | 4 | −1 |
| MC73-D | 9 | 12 | +3 |
| RC73-A-2.5 | 352 ᵇ | 275 | −77 |
| RC73-A-3.1 | 87 | 82 | −5 |
| RC73-B-2.5 | 276 ᵇ | 236 | −40 |
| RC73-B-3.1 | 67 | 65 | −2 |

ᵃ Crecimiento que la investigadora consideró no válido para el conteo, por lo que las detecciones del sistema son falsos positivos. ᵇ Por encima del rango contable estándar de 250 colonias (TNTC).

Las Figuras 1.3 a 1.17 muestran, para cada una de las 15 placas, el conteo manual de referencia sobre la fotografía original junto con la detección automática de CellSAM sobre la placa recortada y corregida. Esta evidencia visual respalda de forma directa los números de la Tabla 1.4 y permite distinguir, placa por placa, si un error se debe a colonias que el detector no propone, a regiones fusionadas o a falsos positivos.

![Figura 1.3. Placa M1-B.](figuras/M1-B_proceso.png)

![Figura 1.4. Placa M1-C.](figuras/M1-C_proceso.png)

![Figura 1.5. Placa M1-D.](figuras/M1-D_proceso.png)

![Figura 1.6. Placa M2-A-3.](figuras/M2-A-3_proceso.png)

![Figura 1.7. Placa M2-A.](figuras/M2-A_proceso.png)

![Figura 1.8. Placa M2-B.](figuras/M2-B_proceso.png)

![Figura 1.9. Placa M2-C.](figuras/M2-C_proceso.png)

![Figura 1.10. Placa MC73-A.](figuras/MC73-A_proceso.png)

![Figura 1.11. Placa MC73-B.](figuras/MC73-B_proceso.png)

![Figura 1.12. Placa MC73-C.](figuras/MC73-C_proceso.png)

![Figura 1.13. Placa MC73-D.](figuras/MC73-D_proceso.png)

![Figura 1.14. Placa RC73-A-2.5.](figuras/RC73-A-2.5_proceso.png)

![Figura 1.15. Placa RC73-A-3.1.](figuras/RC73-A-3.1_proceso.png)

![Figura 1.16. Placa RC73-B-2.5.](figuras/RC73-B-2.5_proceso.png)

![Figura 1.17. Placa RC73-B-3.1.](figuras/RC73-B-3.1_proceso.png)

**Contraste con el conteo manual de referencia.** La propia investigadora realizó el conteo manual de las 15 placas con una herramienta de anotación por clic que registra las coordenadas de cada colonia marcada, lo que permite auditar el conteo y deja abierta la evaluación de localización en el futuro. Sobre las 13 placas dentro del rango contable estándar, el sistema alcanzó un MAE de 4.08 colonias/placa, mejor incluso que el MAE de 5.94 del conjunto de referencia original, con coincidencia exacta en M1-B y M2-B y con errores de dos a tres colonias en la mayor parte de las series MC73 y RC73. Sobre el total de 856 colonias contadas a mano, el acierto agregado es del 80.1 %. Si se consideran las 15 placas, el MAE global sube a 11.33, debido sobre todo a las dos placas que superan el rango contable, RC73-A-2.5 (352 colonias manuales frente a 275 del sistema) y RC73-B-2.5 (276 frente a 236), en las que la fusión de colonias vecinas produce un subconteo de entre el 14 y el 22 %. En la microbiología convencional, las placas con más de 250 colonias se registran como incontables (TNTC), así que ese caso queda fuera del alcance previsto del sistema.

El radio de recorte de la placa se fijó en el 92 % tras medir, sobre el propio conteo manual, qué fracción de las colonias reales queda dentro del área analizada según ese parámetro. La curva de recuperación muestra que un recorte al 86 % deja fuera el 21 % de las 856 colonias marcadas, mientras que al 92 % la cobertura sube al 96 %. La comparación contra el conteo manual confirma la mejora, ya que el MAE en rango contable baja de 5.46 a 4.08 y el acierto agregado sube del 73.0 % al 80.1 %. La ganancia se concentra en las placas densas, donde RC73-A-2.5 reduce su error de 109 a 77 colonias. El ajuste tiene un costo medible, porque MC73-A y MC73-D dejan de coincidir de forma exacta con el conteo manual (13 a 16 y 9 a 12) por falsos positivos de condensación que entran al ampliar el área analizada. Ampliar hasta el radio completo resulta contraproducente, con un MAE en rango contable de 5.77, porque la condensación adherida a la pared de la placa forma gotas redondas que ningún filtro de forma logra separar de una colonia real.

Los errores restantes se concentran en dos modos de fallo con signo opuesto. El primero es el subconteo de colonias pequeñas y pálidas, que se observa sobre todo en M1-D, donde el conteo manual registró 21 colonias y el sistema solo 8. El segundo son los falsos positivos, con M2-A-3 como caso extremo, porque el conteo manual de esa placa es 0 por criterio explícito de la investigadora, que consideró el crecimiento en cadena como no válido para el conteo, y por eso las 11 detecciones del sistema son falsas; algo parecido ocurre, en menor medida, en M2-A, con 9 detecciones frente a 3 colonias manuales, y en M2-C, con 6 frente a 2. Este caso obliga a acotar el alcance de la corrección de iluminación, ya que la corrección hace visibles las estructuras pálidas pero no distingue entre colonias válidas y crecimiento no contable, y esa distinción hoy solo la aporta el criterio experto.

**El selector adaptativo de umbral no funciona en estas condiciones.** El módulo de estimación de densidad, que sí funciona sobre el conjunto de referencia, no se transfiere a este montaje de captura, porque sus estimaciones no guardan relación con la densidad real, hasta el punto de asignar una estimación de 10 a la placa RC73-A-2.5, que contiene 352 colonias según el conteo manual, y de 123 a la placa M1-B, que contiene 2. La relación llega a ser inversa a la esperada, ya que la placa más poblada del conjunto recibe una de las estimaciones más bajas. En la práctica, las 15 placas se procesaron con un umbral fijo de 0.40, con la única excepción de M1-B, que fue clasificada de forma errónea como densa y procesada con 0.65. El módulo requiere una recalibración con fotografías de este montaje antes de poder usarse aquí.

**Fragilidad del filtro de color ante el balance de blancos.** El rechazo de la rotulación se apoya en un rango fijo de matiz, de 60 a 140, medido sobre el primer conjunto de fotografías, donde las colonias quedaban en 39 a 44 y la tinta en 71 a 105. En un segundo conjunto de fotografías, tomadas con el mismo montaje semanas despues, el balance de color de la cámara desplazó el matiz de las colonias a una mediana de 56 con percentil 90 en 66, es decir al borde del umbral. El filtro conservó su margen por poco, pero una variación algo mayor habría empezado a descartar colonias reales. Se ensayó una versión adaptativa que mide el matiz del agar en cada fotografía y rechaza lo que se aparta de él, con resultados equivalentes sobre el primer conjunto (MAE 4.08 frente a 4.00), como era esperable dado que el rango fijo se calibró justamente sobre esas imágenes. Su ventaja debería manifestarse en conjuntos con otro balance de color, lo que queda pendiente de comprobación.

**El límite está en el detector, no en el filtrado.** También se documentó un resultado negativo que fue útil descartar, porque se sospechaba que el filtro de área mínima de 300 px² eliminaba colonias pequeñas ya detectadas, pero la medición mostró que la región más pequeña propuesta por el modelo tenía 648 px², muy por encima de ese umbral. En consecuencia, las colonias faltantes no se pierden en el filtrado, sino que el detector no las propone, y la mejora del caso de colonias pequeñas y pálidas exige actuar sobre el modelo o sobre la imagen de entrada, no sobre los filtros posteriores.

---

## 1.5 Objetivos

### 1.5.1 Objetivo General

Desarrollar y validar un sistema automatizado de conteo de colonias de actinomicetos en imágenes macroscópicas de placas de Petri capturadas con dispositivos de consumo masivo, basado en el modelo de fundación CellSAM con preprocesamiento adaptativo CLAHE y selección adaptativa del umbral de detección, y diseñado para su despliegue como aplicación web accesible para investigadores en campo.

### 1.5.2 Objetivos Específicos

- Implementar un pipeline de preprocesamiento basado en CLAHE que permita la detección de colonias de actinomicetos por CellSAM en imágenes macroscópicas de placa capturadas con teléfono inteligente.

- Optimizar el parámetro `bbox_threshold` de CellSAM mediante un barrido sistemático de valores (0.10–0.80) y diseñar un módulo de selección adaptativa del umbral basado en la estimación de densidad de colonias por visión clásica (top-hat + análisis de blobs).

- Evaluar la precisión del sistema frente a conteos manuales de referencia (*ground truth*) sobre un conjunto de 8 imágenes de placas de actinomicetos (16 placas en total), utilizando el error absoluto medio (MAE) como métrica principal, con un objetivo de MAE < 10 colonias/placa.

- Validar la transferibilidad del pipeline sobre un conjunto de 15 placas sembradas y fotografiadas por la propia investigadora en condiciones de captura distintas (transiluminador, una placa por imagen, rotulación con marcador en el borde), incorporando una corrección de iluminación *flat-field* y filtros de artefactos, y evaluando la precisión frente a un conteo manual de referencia construido por la propia investigadora mediante anotación por clic con registro de coordenadas.

- Diseñar la arquitectura de una aplicación web que permita a los investigadores cargar fotografías de placas tomadas con teléfono inteligente y obtener el conteo automático de colonias con la visualización de las máscaras de segmentación por instancia.

---

## Referencias Bibliográficas

Bérdy, J. "Bioactive microbial metabolites." *The Journal of Antibiotics*, vol. 58, no. 1, pp. 1–26, 2005. https://doi.org/10.1038/ja.2005.1

Watve, M.G., Tickoo, R., Jog, M.M., & Bhole, B.D. "How many antibiotics are produced by the genus *Streptomyces*?" *Archives of Microbiology*, vol. 176, no. 5, pp. 386–390, 2001. https://doi.org/10.1007/s002030100345

Murray, C.J.L., et al. "Global burden of bacterial antimicrobial resistance in 2019: a systematic analysis." *The Lancet*, vol. 399, pp. 629–655, 2022. https://doi.org/10.1016/S0140-6736(21)02724-0

O'Neill, J. "Tackling Drug-Resistant Infections Globally: Final Report and Recommendations." *The Review on Antimicrobial Resistance*, Londres, 2016.

Tiwari, K., & Gupta, R.K. "Rare actinomycetes: a potential storehouse for novel antibiotics." *Critical Reviews in Biotechnology*, vol. 32, no. 2, pp. 108–132, 2012. https://doi.org/10.3109/07388551.2011.562482

Adegboye, M.F., & Babalola, O.O. "Taxonomy and ecology of antibiotic producing actinomycetes." *African Journal of Agricultural Research*, vol. 7, no. 15, pp. 2255–2261, 2012. https://doi.org/10.5897/AJARX11.071

Cutler, K.J., Stringer, C., Lo, T.W., Rappez, L., Stroustrup, N., Peterson, S.B., et al. "Omnipose: a high-precision morphology-independent solution for bacterial cell segmentation." *Nature Methods*, vol. 19, pp. 1438–1448, 2022. https://doi.org/10.1038/s41592-022-01639-4

Marks, M., Israel, U., Dilip, R., et al. "CellSAM: a foundation model for cell segmentation." *Nature Methods*, vol. 22, no. 12, pp. 2585–2593, 2025. https://doi.org/10.1038/s41592-025-02879-w

Hallström, E., Kandavalli, V., Ranefall, P., Elf, J., Wählby, C. "Label-free deep learning-based species classification of bacteria imaged by phase-contrast microscopy." *PLoS Computational Biology*, vol. 19, no. 11, e1011181, 2023. https://doi.org/10.1371/journal.pcbi.1011181

Hallström, E., Kandavalli, V., Wählby, C., Hast, A. "Rapid label-free identification of seven bacterial species using microfluidics, single-cell time-lapse phase-contrast microscopy, and deep learning-based image and video classification." *PLOS ONE*, vol. 20, no. 9, e0330265, 2025. https://doi.org/10.1371/journal.pone.0330265

Ikebe, M., Aoki, K., Hayashi-Nishino, M., Furusawa, C., Nishino, K. "Bioinformatic analysis reveals the association between bacterial morphology and antibiotic resistance using light microscopy with deep learning." *Frontiers in Microbiology*, vol. 15, 1450804, 2024. https://doi.org/10.3389/fmicb.2024.1450804

Szabó, S., Feier, B., Capatina, D., Tertis, M., Cristea, C., & Popa, A. "An Overview of Healthcare Associated Infections and Their Detection Methods Caused by Pathogen Bacteria in Romania and Europe." *Journal of Clinical Medicine*, vol. 11, no. 11, 3204, 2022. https://doi.org/10.3390/jcm11113204

Acuña Quiñones, D.J. "Automatización del conteo de *Artemia salina* en estadio I-III y cistos con visión artificial asistido por un posicionador bidimensional controlado desde una aplicación web." [Universidad, 2023]. [Enlace Cybertesis, PENDIENTE]

Putman, M., Burton, R., & Nahm, M.H. "Simplified method to automatically count bacterial colony forming unit." *Journal of Immunological Methods*, vol. 302, no. 1–2, pp. 99–102, 2005. https://doi.org/10.1016/j.jim.2005.05.003

Clarke, M.L., Burton, R.L., Hill, A.N., Litorja, M., Nahm, M.H., & Hwang, J. "Low-cost, high-throughput, automated counting of bacterial colonies." *Cytometry Part A*, vol. 77A, no. 8, pp. 790–797, 2010. https://doi.org/10.1002/cyto.a.20864

Brugger, S.D., Baumberger, C., Jost, M., Jenni, W., Brugger, U., & Mühlemann, K. "Automated counting of bacterial colony forming units on agar plates." *PLoS ONE*, vol. 7, no. 3, e33695, 2012. https://doi.org/10.1371/journal.pone.0033695

---

> **Notas para completar:**
> - **Figuras 1.1 y 1.2:** generar con datos de Scopus (búsqueda: *Actinomycetes* y *Actinomycetes AND automated counting*, 1990–2024) y registrar la fecha de consulta y la sintaxis exacta, para poder citar las cifras del primer párrafo de 1.4 como recuento verificable.
> - **Acuña (2023):** completar datos de universidad y enlace Cybertesis.
> - **Costos de las Tablas 1.1 y 1.2:** los rangos de precio del conteo manual, de los contadores comerciales y del equipo MALDI-TOF provienen de estimaciones de mercado y no de las referencias citadas. Conviene respaldarlos con cotizaciones o catálogos de proveedor, o presentarlos explícitamente como estimaciones propias.

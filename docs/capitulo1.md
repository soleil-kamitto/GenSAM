# CAPÍTULO 1

---

## 1.1 Introducción

Los actinomicetos son uno de los grupos bacterianos con mayor importancia biotecnológica a nivel mundial, ya que estas bacterias Gram-positivas filamentosas, presentes sobre todo en los suelos, se encuentran entre las principales fuentes de antibióticos de origen natural. Cerca del 45 % de los metabolitos microbianos bioactivos conocidos proviene de actinomicetos (Bérdy, 2005), y se estima que el género *Streptomyces* por sí solo produce alrededor del 75 % de los antibióticos de utilidad clínica y comercial (Watve et al., 2001), entre ellos la estreptomicina, la eritromicina, la tetraciclina y la vancomicina. La resistencia antimicrobiana fue responsable directa de cerca de 1,27 millones de muertes en el mundo durante 2019 (Murray et al., 2022), y podría llegar a 10 millones de muertes anuales hacia 2050 si las tendencias actuales se mantienen (O'Neill, 2016), de modo que la búsqueda sistemática de nuevas cepas de actinomicetos con actividad biológica es hoy una prioridad de investigación. Esta búsqueda depende del cultivo en placas de Petri y del conteo de las colonias características, porque a mayor número de muestras analizadas, mayor es la probabilidad de encontrar cepas que produzcan compuestos bioactivos nuevos (Tiwari & Gupta, 2012).

El procedimiento estándar en los laboratorios de microbiología ambiental consiste en sembrar diluciones de muestras de suelo en agar selectivo, incubar las placas durante 7 a 14 días y contar de forma manual las colonias características de actinomicetos, que se reconocen por su superficie polvosa o dura, su color variable y su aspecto esporulado. En los programas de prospección de gran escala, cada muestra de suelo se siembra en varias diluciones y réplicas sobre medios selectivos (Adegboye & Babalola, 2012), de modo que una sola campaña de muestreo puede generar decenas o cientos de placas, y cada una exige una inspección visual cuidadosa cuya duración crece con la densidad de colonias. Este cuello de botella limita de manera importante la capacidad de cribado de las instituciones de investigación.

En los últimos años, la combinación de modelos de segmentación profunda con grandes colecciones de imágenes biológicas ha producido herramientas de propósito general para el análisis de imágenes de células. Cutler et al. (2022) desarrollaron Omnipose, una red neuronal basada en U-net que supera a los métodos clásicos de umbralización en la segmentación precisa de células bacterianas de forma arbitraria, y más recientemente Marks et al. (2025) publicaron CellSAM, un modelo de fundación (*foundation model*) construido sobre la arquitectura SAM (*Segment Anything Model*) y el detector AnchorDETR, entrenado con un corpus diverso de imágenes de células de múltiples modalidades de microscopía. CellSAM muestra capacidad de generalización hacia tipos de células que no estaban en su entrenamiento original (Marks et al., 2025), y por esa razón es un buen candidato para adaptarlo a dominios nuevos si la imagen de entrada se prepara de forma adecuada.

Ahora bien, la aplicación de estos modelos al conteo de colonias sobre fotografías de teléfono ya ha sido explorada, y con buenos resultados. Existen trabajos que combinan modelos de fundación para detectar y segmentar colonias sin entrenamiento adicional (Colony Grounded SAM2, 2026), y otros que resuelven la pérdida de colonias pequeñas mediante inferencia por baldosas (Yildiz et al., 2026). Por lo tanto, la aportación de este trabajo no puede ser el método en sí mismo, y conviene decirlo desde el principio. Lo que sigue sin estar demostrado, y lo que esos mismos autores señalan como pendiente, es que un contador ajustado en un laboratorio siga funcionando en otro, con otro teléfono, otra iluminación y otro medio de cultivo. Este trabajo aborda ese problema, lo mide y propone una solución, y usa como caso de estudio el conteo de colonias de actinomicetos en fotografías macroscópicas tomadas con equipos comunes.

---

## 1.2 Descripción del Problema

El conteo manual de colonias de actinomicetos en placas de Petri es hoy el procedimiento estándar en la mayoría de los laboratorios de microbiología ambiental y farmacéutica. En este procedimiento, el operador inspecciona cada placa bajo iluminación directa, identifica las colonias características por su forma y su color, y las cuenta una por una. Este método presenta cuatro limitaciones principales.

**Tiempo operativo elevado.** El tiempo de inspección de cada placa crece con la densidad de colonias y depende de la experiencia del operador, y como los protocolos de aislamiento siembran cada muestra en varias diluciones y réplicas sobre medios selectivos (Adegboye & Babalola, 2012), el conteo manual acumulado de una campaña de prospección termina ocupando una fracción desproporcionada del tiempo total de la investigación.

**Variabilidad entre observadores y tendencia al error.** Aunque el conteo manual sigue siendo el estándar de referencia, es lento y produce resultados distintos entre observadores, porque cada uno aplica su propio criterio para incluir las colonias marginales, superpuestas o difíciles de ver (Putman et al., 2005; Clarke et al., 2010). Además, en comparaciones directas contra un estándar de referencia, el conteo manual de rutina mostró desviaciones sistemáticas de subconteo, que fueron más marcadas en las placas de alta densidad (Brugger et al., 2012).

**Fatiga visual y error acumulado.** El conteo repetitivo es una tarea monótona y propensa al error (Clarke et al., 2010), de modo que las equivocaciones tienden a aumentar con el número de placas procesadas en una sesión y los resultados de un mismo estudio se vuelven menos comparables entre sí.

**Escalabilidad limitada.** Como el procedimiento depende de un operador humano, existe un techo operativo que hace inviable el análisis de grandes colecciones de muestras en plazos razonables, y ese techo limita el alcance de los programas de prospección de nuevos antibióticos (Tiwari & Gupta, 2012).

El problema se agrava por la forma propia de las colonias de actinomicetos, ya que presentan bordes difusos, textura granular o polvosa, colores variados (blanco, gris, naranja, rojo o marrón) y tendencia a superponerse en las placas de alta densidad. Las herramientas de análisis de imagen disponibles, como los plugins de conteo de colonias en ImageJ/Fiji o los contadores automáticos comerciales basados en umbralización clásica, fueron diseñadas y validadas para colonias compactas de bordes bien definidos sobre medios de contraste uniforme, y sus propios autores reconocen que las colonias de apariencia muy distinta o de bordes difusos exigen adaptar el algoritmo (Clarke et al., 2010; Brugger et al., 2012), que es justo el caso de los actinomicetos.

### 1.2.1 El problema que este trabajo aborda: la transferibilidad

Los modelos de segmentación de células de última generación, como Omnipose (Cutler et al., 2022) y CellSAM (Marks et al., 2025), fueron entrenados y validados sobre todo con imágenes de microscopía, cuyas condiciones de contraste, escala y fondo son muy distintas a las de una placa macroscópica fotografiada con celular. En los experimentos de este trabajo, la aplicación directa de CellSAM sobre imágenes de placa sin preprocesamiento produjo cero detecciones, porque el modelo no distingue las colonias del fondo del agar sin una normalización de contraste previa. Ese primer obstáculo se resuelve con una ecualización adaptativa del histograma, y una vez resuelto aparece un problema de fondo que resulta bastante más difícil.

Un sistema de conteo automático no consiste solo en un modelo, sino en un modelo rodeado de decisiones: dónde recortar la placa, cuánto contraste añadir, qué umbral de detección usar, qué regiones descartar por tamaño y cuáles por color. Cada una de esas decisiones se toma mirando unas imágenes concretas, y por eso cada una queda ligada a las condiciones en que esas imágenes se tomaron. Mientras el sistema se evalúe sobre fotografías del mismo montaje, esa dependencia permanece invisible, y el sistema parece funcionar bien.

El problema aparece cuando el sistema cambia de laboratorio. En este trabajo se midieron **cuatro parámetros distintos que dejan de funcionar al cambiar el montaje de captura**, y el detalle de cada caso se presenta en la sección 1.4.2. Además se encontraron **dos errores de programa que solo se manifiestan fuera del laboratorio de origen**, y que por su naturaleza no borran el resultado sino que lo alteran en silencio, de modo que un usuario sin conteo manual propio no tendría manera de notarlos.

Esta situación es especialmente grave para el escenario que motiva el trabajo, porque un laboratorio sin presupuesto no dispone de un conteo manual exhaustivo con el que verificar la herramienta que acaba de instalar. Si la herramienta necesita recalibración y no lo advierte, ese laboratorio obtendrá cifras equivocadas y las creerá correctas. Por lo tanto, la pregunta que este trabajo plantea no es solo si un contador automático puede funcionar con fotografías de teléfono, sino si puede funcionar **sin recalibrarse** en un laboratorio que no participó en su desarrollo.

---

## 1.3 Justificación

**Urgencia del descubrimiento de nuevos antibióticos.** La resistencia antimicrobiana es una de las mayores amenazas para la salud pública de este siglo, ya que fue responsable directa de 1,27 millones de muertes en 2019 (Murray et al., 2022) y podría llegar a 10 millones de muertes anuales hacia 2050 si las tendencias actuales continúan (O'Neill, 2016). Los actinomicetos, y en particular el género *Streptomyces*, han sido la fuente de la mayor parte de los antibióticos de origen natural que hoy se usan en la clínica (Watve et al., 2001; Bérdy, 2005). Sin embargo, el descubrimiento de compuestos bioactivos nuevos se ha vuelto más lento, y encontrar cepas productoras exige aislar y cribar volúmenes crecientes de muestras (Tiwari & Gupta, 2012), de modo que el conteo manual de colonias se convierte en un cuello de botella. Un sistema que automatice este paso libera una parte importante del tiempo del investigador y amplía el espacio de búsqueda de cepas nuevas.

**Accesibilidad tecnológica para laboratorios de recursos limitados.** Las soluciones basadas en microscopía de contraste de fase requieren equipos especializados de alto costo, y la espectrometría MALDI-TOF, aunque se ha consolidado como un método rápido de identificación bacteriana (Szabó et al., 2022), exige una inversión inicial del orden de cientos de miles de dólares. A diferencia de ambas, el presente sistema trabaja con fotografías macroscópicas de placas de Petri tomadas con el teléfono del investigador, así que se elimina la barrera del equipamiento y el despliegue se vuelve viable en laboratorios universitarios, institutos de investigación regionales y programas de monitoreo ambiental en campo, contextos que son especialmente relevantes en el Perú y en América Latina (Acuña, 2023).

**La transferibilidad como aportación central.** Varios trabajos recientes han demostrado que el conteo automático sobre fotografías de teléfono es viable, y lo han hecho con métricas altas (Yildiz et al., 2026; Colony Grounded SAM2, 2026). Sin embargo, todos ellos se evalúan sobre un único conjunto de imágenes, y sus autores señalan de forma explícita que falta la validación externa en distintos laboratorios y dispositivos. Este trabajo se sitúa justamente en ese hueco, porque no se limita a proponer un método sino que mide en qué condiciones deja de funcionar, documenta los modos concretos de fallo y sustituye las constantes ajustadas a mano por reglas que se derivan de cada fotografía. Esa diferencia importa para el usuario final, ya que un laboratorio sin presupuesto no puede recalibrar lo que instala, y tampoco puede detectar que necesitaría hacerlo.

**Viabilidad técnica demostrada.** En este trabajo, CellSAM muestra capacidad de generalización hacia imágenes macroscópicas de actinomicetos con una modificación que no toca el modelo, porque solo se añade una etapa de preprocesamiento que corrige el bajo contraste entre las colonias y el agar. Cuando esta etapa se complementa con una selección del umbral de detección derivada de la propia imagen, el sistema alcanza un MAE de **5,94 colonias por placa** sobre el conjunto de referencia con conteo manual, formado por 8 imágenes con dos placas cada una, y esa cifra representa una reducción del 53 % frente a la configuración base, cuyo MAE era de 12,75. Sobre las fotografías propias, y una vez excluidos los cultivos fallidos según el criterio biológico que se detalla en 1.4.1, el MAE baja a **2,83 colonias por placa** con un acierto agregado del 83,8 %. Estos resultados demuestran la viabilidad del enfoque sin necesidad de entrenar el modelo desde cero.

**Rigor metodológico: resultados negativos y pruebas a ciegas.** El desarrollo incorporó dos prácticas que conviene destacar porque sostienen la credibilidad de las cifras anteriores. La primera es el uso sistemático de **controles negativos**, es decir, conjuntos donde una regla propuesta no debería activarse, y esa práctica sirvió para descartar al menos una regla que parecía perfecta sobre los datos de desarrollo. La segunda es que el tercer lote de placas se procesó **a ciegas**, porque la investigadora conservó su conteo manual sin comunicarlo mientras se ajustaba el sistema, de modo que ninguna decisión pudo tomarse mirando el resultado. Ambas prácticas son poco frecuentes en la literatura de conteo de colonias, y las dos se documentan en detalle en la sección 1.4.

**Código abierto y escalabilidad.** El sistema está implementado por completo en Python con bibliotecas de código abierto (PyTorch, OpenCV, scikit-image), de modo que la comunidad científica puede auditarlo, reproducirlo y extenderlo. Además, el diseño como servicio web facilita su integración en los flujos de trabajo existentes sin instalar dependencias complejas, porque el investigador sube sus fotografías desde el celular y recibe en segundos el conteo automático con la imagen segmentada.

---

## 1.4 Estado del Arte

Los actinomicetos acumulan un volumen creciente de investigación científica, ya que en la base Scopus aparecen actualmente cerca de 45.000 publicaciones con el término *Actinomycetes* y más de 120.000 con *Streptomyces* (cifras aproximadas, que dependen de la sintaxis de búsqueda empleada), con aumentos sostenidos desde la década de 1990 impulsados por la búsqueda de compuestos bioactivos nuevos. Sin embargo, dentro de ese volumen, los trabajos dedicados de forma específica a la **automatización del conteo de colonias mediante visión artificial** siguen siendo escasos, como se ilustra en las Figuras 1.1 y 1.2.

> **[Figura 1.1, PENDIENTE]** Número de publicaciones por año en Scopus con keyword: *Actinomycetes* (1990–2024).
>
> **[Figura 1.2, PENDIENTE]** Número de publicaciones por año en Scopus con keywords: *Actinomycetes* AND *automated counting* (1990–2024).

Para poner en contexto el panorama tecnológico actual, la **Tabla 1.1** presenta los métodos convencionales de conteo de colonias que se emplean en los laboratorios de microbiología, y la **Tabla 1.2** resume las soluciones basadas en visión artificial y aprendizaje profundo que han aparecido en los últimos cinco años.

**Tabla 1.1. Métodos convencionales para el conteo de colonias en placas de Petri.**

| N° | Tecnología | Descripción | Costo estimado | Referencia |
|----|------------|-------------|----------------|------------|
| 1 | Conteo manual (visual) | Cuantificación visual por operador bajo iluminación directa o lupa. Sujeto a fatiga, lentitud y variabilidad entre observadores. | $15–30/hr (labor del operador) | Putman et al. (2005); Clarke et al. (2010) |
| 2 | Contadores automáticos comerciales (p.ej. aCOLyte, ProtoCOL) | Imagen de placa + software de umbralización clásica. Limitado a colonias de forma regular en medios claros. | $3.000–$8.000 (hardware + licencia) | Clarke et al. (2010); Brugger et al. (2012) |
| 3 | MALDI-TOF Spectrometry | Estándar clínico de identificación por perfil proteico. No realiza conteo de colonias, porque se usa en la identificación de especie posterior al aislamiento. | $150.000–$250.000 (adquisición inicial) | Szabó et al. (2022) |

**Tabla 1.2. Soluciones basadas en visión artificial y aprendizaje profundo para segmentación y conteo celular.**

| N° | Tecnología | Descripción | Costo aproximado | Referencia |
|----|------------|-------------|-----------------|------------|
| 1 | Omnipose (Deep Learning) | Red U-net con campo de gradiente de función de distancia. Alta precisión en células bacterianas de forma arbitraria en microscopía, y supera a los métodos clásicos en cultivos mixtos y formas alargadas o ramificadas. | $0 software + GPU | Cutler et al. (2022) |
| 2 | CellSAM (Foundation Model) | Modelo de fundación basado en SAM + AnchorDETR, entrenado con un corpus diverso de imágenes celulares de múltiples modalidades. Capacidad de generalización hacia tipos celulares no vistos en el entrenamiento. | $0 software + CPU/GPU | Marks et al. (2025) |
| 3 | ResNet-50 / Vision Transformer (ViT) | Clasificación automática de cepas resistentes a antibióticos mediante cambios de forma cuantificables en microscopía de luz, sin requerir pruebas bioquímicas adicionales. | No estimado | Ikebe et al. (2024) |
| 4 | YOLO con inferencia por baldosas (SAHI) | Trocea la fotografía en baldosas solapadas de 640 × 640 antes de detectar, para que las colonias pequeñas no desaparezcan al reducir la imagen. Sobre fotografías de tres teléfonos distintos, el mAP@0.5 sube del rango 44,9–66,3 % al rango 95,4–96,9 %. | $0 software + GPU | Yildiz et al. (2026) |
| 5 | Colony Grounded SAM2 | Combina Grounding DINO con SAM 2 para detectar y segmentar colonias prácticamente sin entrenamiento, con 93,1 % de precisión media sobre el conjunto ADBC. No trata organismos filamentosos. | $0 software + GPU | Colony Grounded SAM2 (2026) |
| 6 | **CellSAM con parámetros derivados de la imagen** *(presente trabajo)* | CellSAM adaptado a imágenes macroscópicas de placas de Petri, con los parámetros del pipeline derivados de propiedades medibles de cada fotografía en lugar de fijados a mano. **MAE = 5,94** en el conjunto de referencia, **MAE = 2,83** y 83,8 % de acierto en las placas propias, con validación externa sobre placas de otro laboratorio. Código abierto. | $0 | Presente trabajo |

Como muestra la Tabla 1.2, el conteo de colonias con modelos de fundación sobre fotografías de teléfono es un terreno ya transitado, y por eso conviene ser preciso sobre dónde queda el aporte de este trabajo. La **Tabla 1.3** contrasta lo que esos trabajos declaran como pendiente con lo que aquí se desarrolla.

**Tabla 1.3. Limitaciones declaradas por los trabajos recientes y su tratamiento en el presente trabajo.**

| Limitación declarada | Fuente | Tratamiento aquí |
|----------------------|--------|------------------|
| Falta validación externa en distintos laboratorios y dispositivos | Yildiz et al. (2026) | Es el objeto central del trabajo, con cuatro parámetros medidos que no transfieren y dos errores que solo aparecen fuera del laboratorio de origen |
| Evaluación sobre un único conjunto público | Yildiz et al. (2026) | Tres lotes propios de captura independiente, más validación sobre el conjunto ADBC |
| Detección por caja en lugar de segmentación de instancias | Yildiz et al. (2026) | CellSAM entrega segmentación de instancias |
| Las colonias densamente agrupadas del centro se pierden | Colony Grounded SAM2 (2026) | Separación por cuencas hidrográficas con criterio de compacidad |
| No se tratan organismos filamentosos | Colony Grounded SAM2 (2026) | Los actinomicetos son el objeto de estudio |

Hallström et al. (2023; 2025) demostraron que la clasificación automática de especies bacterianas con aprendizaje profundo sobre imágenes de microscopía es viable con precisiones superiores al 93 %, aunque sus enfoques requieren chips microfluídicos y microscopios de contraste de fase, por lo que su uso queda limitado a laboratorios muy equipados. En el contexto latinoamericano, Acuña (2023) demostró que es viable construir sistemas de visión artificial con hardware impreso en 3D y control web para el conteo automático de microorganismos acuáticos, y ese antecedente metodológico se aplica de forma directa al presente trabajo.

En cuanto a conjuntos de datos anotados, Rodríguez et al. (2023) publicaron ADBC, que reúne 369 fotografías de placas de 24 especies bacterianas con 56.865 colonias anotadas, tomadas con tres teléfonos distintos y sin iluminación normalizada ni protocolo estricto de posicionamiento. Por su parte, Majchrowska et al. (2021) publicaron AGAR, con 18.000 imágenes y 336.442 colonias. Ambos conjuntos se emplean aquí como referencia externa, y ADBC en particular sirve como prueba de transferencia porque sus condiciones de captura no tienen nada que ver con las del laboratorio de origen.

### 1.4.1 Resultados sobre placas propias

Además del conjunto de referencia con conteo manual, el pipeline se aplicó a fotografías de placas sembradas por la propia investigadora, y esa aplicación es la primera prueba sobre imágenes generadas por completo dentro del proyecto y con un montaje de captura distinto. El material propio se organiza en tres lotes, cuya composición se resume en la **Tabla 1.4**.

**Tabla 1.4. Conjuntos de imágenes empleados en el trabajo.**

| Conjunto | Placas | Colonias | Condiciones de captura |
|----------|-------:|---------:|------------------------|
| Referencia, `images/placas` | 16 | 819 | Dos placas por imagen, iluminación frontal |
| Primer lote propio | 15 | 856 | Transiluminador, contraluz, rotulación azul en el borde |
| Segundo lote propio | 4 | sin contar | Mismo montaje, semanas después |
| Tercer lote propio | 10 | prueba a ciegas | Mejor iluminación, rotulación negra abundante |
| ADBC (externo) | 369 | 56.865 | Tres teléfonos, 24 especies, sin protocolo |
| Sintéticas generadas | 1.000 | 71.522 | Generadas a partir de parches reales |

Las placas del primer lote corresponden a las series M1, M2, MC73 y RC73, con réplicas identificadas por letra y, en algunos casos, con la dilución indicada en el nombre, y la captura se realizó con cámara de teléfono a una resolución de hasta 4096 px, con la placa colocada sobre un transiluminador que la ilumina a contraluz, una placa por imagen y la rotulación escrita con marcador azul sobre el borde.

Estas condiciones traen tres dificultades que no estaban en el conjunto de referencia. La primera es un gradiente de iluminación pronunciado, propio del contraluz, que oscurece la periferia de la placa respecto del centro; la segunda es la rotulación con marcador dentro del campo de la imagen, que el segmentador tiende a proponer como si fuera una colonia; y la tercera es la banda de artefactos del borde de la placa, donde se juntan el menisco del agar y la condensación. El pipeline adaptado responde a las tres dificultades. Primero detecta la placa mediante la transformada de Hough y recorta al 92 % del radio, un valor que se fijó con la curva de recuperación descrita más abajo, y luego reescala el recorte a 1200 px, que es la escala con la que se calibró el pipeline original. Después aplica la corrección de iluminación de campo plano, que consiste en dividir la imagen por un fondo estimado y aplicar después una ecualización adaptativa del histograma. Por último ejecuta CellSAM y descarta regiones por área, por solidez y por color.

La corrección de iluminación resultó ser el hallazgo decisivo para este escenario, y el experimento factorial que la sustenta se resume en la **Tabla 1.5**. En la placa M2-A-3, la más afectada por el contraluz, el sistema no proponía ninguna región sin corrección, y tampoco lo hacía al bajar el umbral, de modo que el problema no era la sensibilidad del detector sino el contraste de la imagen de entrada. Sin embargo, el conteo manual posterior obligó a revisar este caso, porque la investigadora consideró no válido el crecimiento de esa placa y esas detecciones resultaron ser falsos positivos. Por lo tanto, la evidencia de la mejora se apoya en MC73-C, donde el conteo subió de 1 a 4 colonias frente a un conteo manual de 5, y en la placa de control MC73-A, que mantuvo su conteo correcto de 13 colonias en las cuatro variantes, lo que confirma que la corrección no daña los casos que ya funcionaban.

**Tabla 1.5. Efecto de la corrección de iluminación y del umbral de detección sobre tres placas propias.**

| Placa | Sin corrección, thr 0.65 | Sin corrección, thr 0.40 | Con corrección, thr 0.65 | Con corrección, thr 0.40 |
|-------|--------------------------|--------------------------|--------------------------|--------------------------|
| M2-A-3 (colonias pálidas en cadena) | 0 | 0 | 15 | **29** |
| MC73-C (fallo parcial) | 1 | 3 | 4 | **4** |
| MC73-A (control de no regresión) | 13 | 13 | 13 | **13** |

El filtro de color aporta una mejora independiente y comprobable, porque sobre una placa estéril la rotulación con marcador generaba 8 detecciones falsas antes de incorporarlo. Su contribución se midió desactivándolo sobre las 15 placas, y sin él el MAE en rango contable sube de 4,00 a 7,31 colonias por placa mientras que el acierto agregado baja del 80,3 % al 75,7 %, de modo que el filtro aporta algo más de cuatro puntos y medio de acierto por sí solo. La **Tabla 1.6** presenta los conteos del sistema frente al conteo manual de referencia de las 15 placas.

**Tabla 1.6. Conteo automático frente a conteo manual de referencia en las 15 placas del primer lote.**

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

**Exclusión de los cultivos fallidos.** La investigadora determinó que en las placas de las series M1 y M2 la densidad del agar impidió el crecimiento colonial, de modo que constituyen cultivos fallidos y no casos de prueba válidos. Son 7 placas con 29 colonias. Al excluirlas, el MAE en rango contable baja de 4,08 a 2,83 colonias por placa y el acierto agregado sube del 80,1 % al 83,8 %. Conviene declarar esta exclusión de forma explícita y con su justificación biológica, porque mejora los resultados de manera sustancial y porque esas placas concentraban los falsos positivos, con 36 de error absoluto acumulado frente a solo 29 colonias reales. Una exclusión de ese peso presentada como filtro silencioso invalidaría la comparación.

Las Figuras 1.3 a 1.17 muestran, para cada una de las 15 placas, el conteo manual de referencia sobre la fotografía original junto con la detección automática de CellSAM sobre la placa recortada y corregida. Esta evidencia visual respalda de forma directa los números de la Tabla 1.6 y permite distinguir, placa por placa, si un error se debe a colonias que el detector no propone, a regiones fusionadas o a falsos positivos.

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

**Contraste con el conteo manual de referencia.** La propia investigadora realizó el conteo manual de las 15 placas con una herramienta de anotación por clic que registra las coordenadas de cada colonia marcada, lo que permite auditar el conteo y deja abierta la evaluación de localización en el futuro. Sobre las 13 placas dentro del rango contable estándar, el sistema alcanzó un MAE de 4,08 colonias por placa, mejor incluso que el MAE de 5,94 del conjunto de referencia original, con coincidencia exacta en M1-B y M2-B y con errores de dos a tres colonias en la mayor parte de las series MC73 y RC73. Si se consideran las 15 placas, el MAE global sube a 11,33, debido sobre todo a las dos placas que superan el rango contable, RC73-A-2.5 con 352 colonias manuales frente a 275 del sistema, y RC73-B-2.5 con 276 frente a 236, en las que la fusión de colonias vecinas produce un subconteo de entre el 14 y el 22 %. En la microbiología convencional, las placas con más de 250 colonias se registran como incontables (TNTC), así que ese caso queda fuera del alcance previsto del sistema.

El radio de recorte de la placa se fijó en el 92 % tras medir, sobre el propio conteo manual, qué fracción de las colonias reales queda dentro del área analizada según ese parámetro. La curva de recuperación muestra que un recorte al 86 % deja fuera el 21 % de las 856 colonias marcadas, mientras que al 92 % la cobertura sube al 96 %. La comparación contra el conteo manual confirma la mejora, ya que el MAE en rango contable baja de 5,46 a 4,08 y el acierto agregado sube del 73,0 % al 80,1 %. La ganancia se concentra en las placas densas, donde RC73-A-2.5 reduce su error de 109 a 77 colonias. El ajuste tiene un costo medible, porque MC73-A y MC73-D dejan de coincidir de forma exacta con el conteo manual por falsos positivos de condensación que entran al ampliar el área analizada. Ampliar hasta el radio completo resulta contraproducente, con un MAE en rango contable de 5,77, porque la condensación adherida a la pared de la placa forma gotas redondas que ningún filtro de forma logra separar de una colonia real.

Los errores restantes se concentran en dos modos de fallo con signo opuesto. El primero es el subconteo de colonias pequeñas y pálidas, que se observa sobre todo en M1-D, donde el conteo manual registró 21 colonias y el sistema solo 8. El segundo son los falsos positivos, con M2-A-3 como caso extremo, porque el conteo manual de esa placa es 0 por criterio explícito de la investigadora. Este caso obliga a acotar el alcance de la corrección de iluminación, ya que la corrección hace visibles las estructuras pálidas pero no distingue entre colonias válidas y crecimiento no contable, y esa distinción hoy solo la aporta el criterio experto.

### 1.4.2 Cuatro parámetros que no transfieren entre montajes de captura

Esta sección reúne el hallazgo central del trabajo. Cada uno de los cuatro casos se detectó al aplicar el sistema a fotografías tomadas en condiciones distintas de aquellas con las que se había ajustado, y los cuatro comparten la misma causa, que es una constante fijada mirando un conjunto concreto de imágenes. La **Tabla 1.7** los resume, y los párrafos siguientes detallan cada caso.

**Tabla 1.7. Parámetros que dejan de funcionar al cambiar el montaje de captura.**

| Parámetro | Ajustado sobre | Cómo falla |
|-----------|----------------|------------|
| Estimador de densidad | Placas dobles del conjunto de referencia | Asigna una estimación de 10 a una placa que tiene 352 colonias, y de 123 a otra que tiene 2 |
| Filtro de color | Primer lote propio | El matiz del agar pasa de 43 a 56 entre lotes, y el filtro queda al borde de descartar colonias reales |
| Umbral de detección | Fotografías con contraluz | Produce un sesgo sistemático de más de ocho colonias por placa sobre el conjunto de referencia |
| Área mínima de colonia | Fotografías propias | Fijada en 300 px², descarta el 28,8 % de las colonias anotadas de ADBC y el 40,6 % en placas densas |

**El estimador de densidad.** El módulo de estimación de densidad, que funciona sobre el conjunto de referencia, no se transfiere al montaje con transiluminador, porque sus estimaciones no guardan relación con la densidad real. La relación llega a ser inversa a la esperada, ya que la placa más poblada del conjunto recibe una de las estimaciones más bajas. En la práctica, las 15 placas del primer lote se procesaron con un umbral fijo, con la única excepción de M1-B, que fue clasificada de forma errónea como densa.

**El filtro de color.** El rechazo de la rotulación se apoyaba en un rango fijo de matiz, medido sobre el primer conjunto de fotografías, donde las colonias quedaban en 39 a 44 y la tinta en 71 a 105. En un segundo conjunto, tomado con el mismo montaje semanas después, el balance de color de la cámara desplazó el matiz de las colonias a una mediana de 56 con percentil 90 en 66, es decir al borde del umbral. El filtro conservó su margen por poco, pero una variación algo mayor habría empezado a descartar colonias reales.

**El umbral de detección.** El selector de umbral recalibrado para las fotografías con contraluz asigna un valor permisivo a casi todas las placas, y al aplicarlo al conjunto de referencia produce un sesgo de más de ocho colonias por placa, con sobreconteo en 14 de las 16 placas. Conviene señalar que la primera hipótesis sobre este sesgo fue errónea y se descartó midiendo, porque se supuso que la causa era la corrección de iluminación, y al ejecutar el conjunto completo sin ella el sesgo resultó idéntico.

**El área mínima de colonia.** El pipeline exigía un área mínima de 300 px², que traducida a unidades reales significa exigir que una colonia mida 1,35 mm. Medido sobre las anotaciones de ADBC, ese valor descartaría el 28,8 % de todas las colonias anotadas, y hasta el 40,6 % en las placas incontables, tal como muestra la **Tabla 1.8**. Si la validación externa se hubiera ejecutado sin corregir esto, habría medido el filtro en lugar del detector.

**Tabla 1.8. Colonias de ADBC que quedarían por debajo del filtro de área original.**

| Colonias por placa | Colonias anotadas | Bajo el filtro de 300 px² |
|--------------------|------------------:|--------------------------:|
| 1 a 10 | 185 | 17,3 % |
| 11 a 30 | 1.014 | 2,1 % |
| 31 a 60 | 2.252 | 5,0 % |
| 61 a 150 | 11.160 | 8,2 % |
| 151 a 250 | 10.532 | 22,9 % |
| Más de 250 | 31.722 | **40,6 %** |

**La solución adoptada.** En los cuatro casos la corrección sigue el mismo principio, que consiste en sustituir la constante por una regla derivada de propiedades medibles de la propia fotografía o de la geometría conocida del problema. El umbral de detección se deriva de cuánto se aparta la imagen de su propio fondo suavizado, el matiz de referencia se mide en el agar de cada fotografía, y el área mínima se declara en milímetros y se traduce a píxeles usando el hecho de que una placa de Petri normalizada mide 90 mm de diámetro. Conviene advertir que la regla del umbral se apoya por ahora en solo dos montajes medidos, de modo que es una hipótesis operativa y no un resultado consolidado, y su validación con fotografías de otros laboratorios queda pendiente.

### 1.4.3 Dos errores que solo se manifiestan fuera del laboratorio de origen

Al aplicar el sistema a las placas de ADBC aparecieron dos defectos de programa que las fotografías propias nunca habrían revelado, y ambos alteran el resultado en silencio en lugar de producir un fallo visible.

**El matiz tratado como número y no como ángulo.** El matiz es un ángulo, y en la escala habitual de la biblioteca OpenCV, que va de 0 a 179, el valor 179 está pegado al 0. Los agares de sangre de ADBC tienen su matiz en 177 a 179, es decir justo en esa costura, de modo que una colonia de matiz 2 dista solo 4 grados del agar pero la resta directa da 176. El filtro la tomaba por rotulación y la descartaba. En las fotografías propias el defecto nunca se manifestó porque su agar está en 41 a 56, lejos de la costura.

La corrección de este defecto resultó más delicada de lo previsto, porque las dos salidas inmediatas fallan, y ambas se comprobaron antes de adoptar ninguna. La mediana no sirve para un ángulo, ya que si el agar reparte sus matices entre 175 y 5 la mediana cae cerca de 90, que es un verde inexistente en la imagen. La moda circular corrige ese caso pero abre otro, porque los píxeles grises y negros reciben matiz cero por convenio y se amontonan en la primera casilla del histograma, hasta el punto de que una placa propia con una zona oscura amplia devolvía 0 frente a 46 de la mediana. Ponderar cada píxel por su saturación corrige ese segundo caso pero abre un tercero, ya que una región pequeña e intensamente coloreada gana entonces a un agar grande y pálido, y una placa doble pasaba a devolver 21, un naranja, frente a 87 de la mediana. La solución que resiste las tres pruebas consiste en descartar los píxeles donde el matiz no está definido y contar por área los que quedan, de modo que gane la superficie mayor y no la más vistosa. Comprobado sobre veinte placas de cuatro procedencias, el estimador resultante coincide con la mediana dentro de un grado en todas las placas propias, y dentro de un mismo montaje resulta más consistente que ella, con 8 grados de dispersión frente a 23.

**El área mínima fijada en píxeles.** Este defecto se describió en la sección anterior como el cuarto parámetro no transferible, y conviene contarlo también aquí porque su efecto es igualmente silencioso, ya que un usuario sin conteo manual propio observaría un subconteo en las placas densas sin ninguna indicación de que la causa está en un filtro y no en el detector.

La enseñanza de ambos casos es la misma. Un sistema puede parecer correcto durante todo su desarrollo y contener defectos que solo se activan en condiciones que el desarrollador no tuvo delante, y la única manera de encontrarlos consiste en probarlo con material ajeno. Esto refuerza el argumento de la sección 1.2.1, porque un laboratorio sin presupuesto no está en condiciones de detectar fallos de este tipo por su cuenta.

### 1.4.4 Prueba a ciegas sobre el tercer lote

El tercer lote de diez placas se procesó bajo un protocolo de prueba a ciegas, ya que la investigadora realizó su conteo manual y lo conservó sin comunicarlo mientras se ajustaba el sistema. De ese modo, ninguna decisión de desarrollo pudo tomarse mirando el resultado, que es la crítica habitual al ajuste de parámetros sobre el conjunto de prueba.

El obstáculo principal de este lote fue la rotulación con marcador, que en las placas más escritas llegaba a generar más regiones propuestas que las propias colonias, hasta 28 frente a 14, de modo que sin tratarla el conteo se duplicaba. Se ensayaron cuatro versiones sucesivas de un filtro por color aplicado después de detectar, y las cuatro comparten el mismo defecto de fondo, porque filtrar a posteriori obliga a decidir sobre regiones que montan a medias sobre la escritura, donde la estadística de color es ambigua por construcción, y porque cuando una colonia toca un trazo el detector las une en una sola región que ya no se puede separar.

La solución consiste en eliminar la tinta antes de segmentar, mediante reconstrucción del área ocupada por el trazo. El procedimiento es legítimo porque la rotulación está escrita sobre el plástico de la placa y no en el agar, así que es una oclusión del recipiente y no parte de la muestra, y porque lo que se reconstruye debajo es agar, cuyo aspecto es liso y predecible. Hubo que añadir una salvaguarda, ya que la reconstrucción dejaba muescas dentadas en el borde de la placa, donde el algoritmo no dispone de vecindario válido del que copiar, y esas muescas añadían una decena de detecciones falsas en una de las placas. La rotulación que toca el borde se excluye entonces del área analizada en lugar de reconstruirse, con el costo de perder una franja estrecha donde de todos modos la escritura impide ver si hay colonias.

Una vez resuelto ese obstáculo, el lote se contó con tres preprocesamientos de fundamento distinto, cuyos resultados aparecen en la **Tabla 1.9**. El primero usa la imagen directa, el segundo aplica densidad óptica según la ley de Beer-Lambert sobre un fondo estimado por morfología, y el tercero trocea la placa en baldosas solapadas a mayor resolución.

**Tabla 1.9. Conteo del tercer lote con tres preprocesamientos distintos.**

| Placa | Imagen directa | Densidad óptica | Baldosas |
|-------|---------------:|----------------:|---------:|
| RC73-1 | 21 | 21 | 19 |
| RC73-2 | 30 | 34 | 30 |
| RC73-3 | 24 | 25 | 24 |
| RC73-4 | 21 | 23 | 23 |
| RC73-5 | 25 | 25 | 25 |
| RC73-6 | 11 | 13 | 13 |
| RC73-7 | 15 | 12 | 13 |
| RC73-8 | 6 | 9 | 5 |
| RC73-9 | 19 | 21 | pendiente |
| RC73-10 | 21 | 21 | 21 |
| **Total** | **193** | **204** | **173 en nueve placas** |

Lo relevante es que los tres métodos convergen. Sobre las nueve placas donde las tres variantes están medidas, los totales son 174, 183 y 173 colonias respectivamente, es decir una dispersión inferior al 6 %, y la densidad óptica se separa de la imagen directa en 1,1 colonias por placa de media mientras que las baldosas se separan en menos de una. Eso permite algo que normalmente exige disponer del conteo manual, porque proporciona una estimación de la incertidumbre del método. Lo que no puede afirmarse a ciegas es cuál de los tres acierta, y por eso la conclusión defendible es más modesta y también más útil, ya que la elección del preprocesamiento no domina el resultado.

### 1.4.5 Resultados negativos documentados

El desarrollo produjo varios resultados negativos, y conviene presentarlos porque delimitan lo que el sistema puede hacer y porque sostienen la credibilidad de los resultados positivos. Dos de ellos merecen mención detallada.

**El descriptor de forma para separar rotulación de colonias.** Las placas del tercer lote están rotuladas con marcador negro, y el negro no tiene matiz, de modo que un criterio basado en el color solo alcanza al 43 % de los píxeles oscuros. Se buscó entonces un criterio independiente del color, y el cociente entre el área de una región y el área del mayor disco inscrito en ella parecía resolverlo, ya que vale 1 para un disco perfecto y crece con el alargamiento. Sobre el tercer lote separaba sin solape, con las colonias entre 1,04 y 1,96 y los trazos entre 3,25 y 16,7.

Antes de adoptarlo se aplicó como control negativo a las placas dobles del conjunto de referencia, que son densas y no tienen escritura alguna, y allí marcó el 70,5 % de los componentes oscuros. El motivo es que en las placas densas las colonias se tocan y forman cadenas, y una cadena resulta tan alargada como un trazo, de modo que la regla habría borrado la mayor parte de las colonias justo en las placas más pobladas. Se ensayaron tres descriptores adicionales para rescatar la idea, que fueron el número de núcleos, el área explicada por discos inscritos y la oscuridad relativa al agar, y las tres distribuciones se solapan casi por completo entre ambos conjuntos. La conclusión es que el color es la única propiedad que separa el marcador de la biomasa, lo cual resulta esperable porque el pigmento del rotulador y el de la colonia son sustancias distintas mientras que su forma y su brillo pueden coincidir. De ahí se sigue una limitación que debe declararse, ya que el sistema trata bien el marcador de color y no puede tratar el marcador negro escrito sobre la zona de cultivo, y su mitigación no es algorítmica sino de protocolo, porque basta con rotular en el reverso o en el anillo exterior.

**El límite está en el detector y no en el filtrado.** Se sospechaba que el filtro de área mínima eliminaba colonias pequeñas ya detectadas, pero la medición mostró que la región más pequeña propuesta por el modelo sobre las placas propias tenía 648 px², muy por encima del umbral. En consecuencia, las colonias faltantes no se pierden en el filtrado sino que el detector no las propone, de modo que mejorar ese caso exige actuar sobre el modelo o sobre la imagen de entrada. Conviene matizar que esta conclusión vale para las placas propias, que son poco pobladas, y que sobre placas densas ajenas el filtro sí resulta determinante, como quedó documentado en la Tabla 1.8.

### 1.4.6 Validación externa sobre ADBC

La validación externa se realiza sobre el conjunto ADBC (Rodríguez et al., 2023), que reúne 369 fotografías de placas de 24 especies bacterianas con 56.865 colonias anotadas, tomadas con tres modelos de teléfono distintos y sin iluminación normalizada. Su composición por densidad aparece en la **Tabla 1.10**.

**Tabla 1.10. Composición del conjunto ADBC por densidad de colonias.**

| Colonias por placa | Placas | Proporción |
|--------------------|-------:|-----------:|
| 1 a 10 | 39 | 10,6 % |
| 11 a 30 | 52 | 14,1 % |
| 31 a 60 | 51 | 13,8 % |
| 61 a 150 | 100 | 27,1 % |
| 151 a 250 | 52 | 14,1 % |
| Más de 250 | 75 | 20,3 % |

La mediana es de 108 colonias por placa y el máximo llega a 747, de modo que se trata de un conjunto bastante más denso que el material propio, donde las placas rondan las 20 colonias, y por lo tanto constituye una prueba exigente y no una confirmación cómoda. Un dato adicional ilustra el desorden de captura que el sistema debe tolerar, porque hay **344 tamaños de imagen distintos en 369 placas**, es decir que casi cada fotografía tiene su propia resolución.

Se dejó registrada una predicción antes de ejecutar la evaluación, para que el resultado pudiera contradecirla. Al reducir la placa a los 1024 píxeles que CellSAM emplea internamente, el diámetro de la colonia mediana pasa de 38,8 px en las placas ralas a 19,3 px en las incontables, y el 57 % de estas últimas queda por debajo del umbral de unos 15 a 20 px donde los detectores pierden fiabilidad. La predicción es que la inferencia por baldosas debe mejorar el conteo en las placas densas y resultar indiferente en las ralas, y que si el resultado sale al revés el razonamiento físico sobre la reducción de escala es incorrecto.

La evaluación se ejecuta sobre una submuestra estratificada con el mismo número de placas por tramo de densidad, elegida con semilla fija, porque no se dispone de unidad de procesamiento gráfico y el modelo tarda minutos por placa sobre procesador. Estratificar importa aquí porque el efecto que se investiga depende justamente de la densidad, de modo que un muestreo al azar dejaría los extremos sin medir. Los resultados de esta evaluación se presentarán en el capítulo correspondiente.

---

## 1.5 Objetivos

### 1.5.1 Objetivo General

Desarrollar y validar un sistema automatizado de conteo de colonias de actinomicetos en imágenes macroscópicas de placas de Petri capturadas con dispositivos de consumo masivo, basado en el modelo de fundación CellSAM, cuyos parámetros se deriven de propiedades medibles de cada fotografía en lugar de fijarse a mano, de modo que pueda transferirse a laboratorios que no participaron en su desarrollo, y diseñado para su despliegue como aplicación web accesible para investigadores en campo.

### 1.5.2 Objetivos Específicos

- Implementar un pipeline de preprocesamiento que permita la detección de colonias de actinomicetos por CellSAM en imágenes macroscópicas de placa capturadas con teléfono inteligente, incorporando corrección de iluminación de campo plano y eliminación de la rotulación previa a la segmentación.

- Identificar y documentar los parámetros del pipeline que no transfieren entre montajes de captura distintos, midiendo en cada caso la magnitud del fallo sobre conjuntos de imágenes tomadas en condiciones diferentes.

- Sustituir esos parámetros por reglas derivadas de propiedades medibles de cada fotografía o de la geometría conocida del problema, y comprobar que las reglas nuevas no degradan el resultado sobre los conjuntos donde las constantes originales funcionaban.

- Evaluar la precisión del sistema frente a conteos manuales de referencia sobre el conjunto de placas dobles y sobre los lotes propios, utilizando como métricas el error absoluto medio y el acierto agregado, y declarando de forma explícita cualquier exclusión de placas junto con su justificación biológica.

- Validar la transferibilidad del sistema sobre el conjunto público ADBC, que reúne 369 placas de 24 especies tomadas con tres teléfonos distintos en otro laboratorio, evaluando el error de conteo por tramo de densidad de colonias.

- Generar un conjunto de placas sintéticas anotadas a partir de parches de colonias reales, con control sobre la densidad, el agrupamiento espacial y las condiciones de captura, que permita entrenar y evaluar detectores sin depender del conteo manual.

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

Yildiz, [iniciales pendientes], et al. "Overcoming resolution constraints in automated colony counting via a high-performance deep learning framework using SAHI." *Scientific Reports*, 2026. https://www.nature.com/articles/s41598-026-55724-1 **[VERIFICAR autoría completa y número de artículo]**

"Colony Grounded SAM2: Zero-shot detection and segmentation of bacterial colonies using foundation models." arXiv:2603.13393, 2026. **[VERIFICAR autoría completa y estado de publicación]**

Rodríguez, [iniciales pendientes], et al. "Annotated dataset for deep-learning-based bacterial colony detection." *Scientific Data*, vol. 10, 2023. https://doi.org/10.1038/s41597-023-02404-8 **[VERIFICAR autoría completa]**

Majchrowska, S., Pawłowski, J., Guła, G., Bonus, T., Hanas, A., Loch, A., et al. "AGAR a microbial colony dataset for deep learning detection." arXiv:2108.01234, 2021.

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
> - **Referencias de 2026 y ADBC:** verificar la autoría completa, el número de artículo y el estado de publicación de Yildiz et al. (2026), de Colony Grounded SAM2 (2026) y de Rodríguez et al. (2023). Las cifras citadas en el texto proceden de los propios artículos, pero los nombres de autor deben confirmarse antes de la entrega.
> - **Figuras 1.1 y 1.2:** generar con datos de Scopus (búsqueda: *Actinomycetes* y *Actinomycetes AND automated counting*, 1990–2024) y registrar la fecha de consulta y la sintaxis exacta, para poder citar las cifras del primer párrafo de 1.4 como recuento verificable.
> - **Acuña (2023):** completar datos de universidad y enlace Cybertesis.
> - **Costos de las Tablas 1.1 y 1.2:** los rangos de precio del conteo manual, de los contadores comerciales y del equipo MALDI-TOF provienen de estimaciones de mercado y no de las referencias citadas. Conviene respaldarlos con cotizaciones o catálogos de proveedor, o presentarlos explícitamente como estimaciones propias.
> - **Sección 1.4.6:** completar con los resultados de la validación sobre ADBC cuando termine la ejecución, y contrastarlos con la predicción registrada.
> - **Tabla 1.9:** completar la columna de baldosas cuando termine esa ejecución.

# Dataset Card · MovieLens

## Fuente, fecha, versión y hash
- **Fuente:** GroupLens Research, Universidad de Minnesota. Conjunto *MovieLens Latest Small* (`ml-latest-small`).
- **URL de descarga:** https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
- **README oficial:** https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
- **Versión:** según su README, generada el 26 de septiembre de 2018. Es un conjunto *de desarrollo*: puede cambiar con el tiempo y GroupLens advierte que no es adecuado para compartir resultados de investigación. Por eso se registran fecha y hash.
- **Fecha de descarga:** 2026-10-15 02:21:30 UTC.
- **SHA-256 del ZIP descargado:** `696d65a3dfceac7c45750ad32df2c259311949efec81f0f144fdfb91ebc9e436`
- **Registro automático:** `reports/dataset_provenance.json` (URL, fecha, hash y conteos).

## Licencia y cita
Condiciones del README oficial:
- Puede usarse para cualquier fin de investigación.
- No se puede declarar ni insinuar el respaldo de la Universidad de Minnesota ni de GroupLens.
- Hay que reconocer el uso del conjunto en las publicaciones resultantes.
- La redistribución solo es posible bajo estas mismas condiciones.
- No puede usarse con fines comerciales o con ingresos sin permiso previo de GroupLens.

En este proyecto **no se redistribuye el dataset**, tal como pide el mandato del curso: `data/raw/` está excluido de Git y se obtiene con el script de descarga.

**Cita:** Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 19:1–19:19. https://doi.org/10.1145/2827872

## Archivos y diccionario
El ZIP contiene `links.csv`, `movies.csv`, `ratings.csv` y `tags.csv`. Este proyecto usa solo los dos primeros que se indican a continuación.

| Archivo | Columna | Significado |
|---|---|---|
| `ratings.csv` | `userId` | Identificador anónimo del usuario |
| | `movieId` | Identificador de la película |
| | `rating` | Valoración de 0.5 a 5.0 estrellas, en pasos de 0.5 |
| | `timestamp` | Segundos desde el 1 de enero de 1970 (UTC) |
| `movies.csv` | `movieId` | Identificador de la película |
| | `title` | Título, normalmente con el año entre paréntesis |
| | `genres` | Géneros separados por `|` |

## Procedimiento de descarga
```bash
uv run python scripts/download_data.py
uv run python scripts/audit_data.py
```
El script descarga el ZIP, calcula el SHA-256 de los bytes recibidos, lo extrae en `data/raw/ml-latest-small/`, valida las columnas y guarda la procedencia en `reports/dataset_provenance.json`.

## Población observada y exclusiones
- 610 usuarios, elegidos al azar entre los usuarios de MovieLens. Todos habían valorado al menos 20 películas.
- 100,836 valoraciones hechas entre el 29 de marzo de 1996 y el 24 de septiembre de 2018.
- 9,742 películas en el catálogo, de las cuales 9,724 tienen al menos una valoración. Las otras 18 no tienen ninguna valoración; el README indica que solo se incluyen películas con al menos una valoración o una etiqueta, así que se deduce que están por sus etiquetas.
- No incluye datos demográficos: cada usuario es solo un identificador.
- Exclusiones aplicadas en LAB08: ninguna. El corte temporal de LAB09 se documenta en la System Card.

## Calidad, sesgos y usos prohibidos
**Calidad y sesgos observados**
- **Densidad del 1.7 %:** solo 100,836 de unos 5.9 millones de pares usuario–película posibles tienen valoración. Un par sin valoración no significa rechazo: la mayoría de las películas nunca se le mostraron o no las vio.
- **Autoselección:** cada persona valora lo que decidió ver. La media es 3.50 y tres cuartas partes de las valoraciones son de 3.0 o más.
- **Cola larga:** cada usuario valoró en promedio 165 películas, pero cada película recibió en promedio solo 10.4 valoraciones. Solo 1,967 películas (20.2 % del catálogo) tienen al menos 12 valoraciones.
- **Títulos con errores:** el README advierte que pueden existir errores e inconsistencias; por ejemplo, la película *Ready Player One* aparece sin año en la consulta por contenido.
- **Géneros gruesos:** cada película tiene una lista corta de géneros de un conjunto fijo, por lo que muchas películas distintas comparten exactamente los mismos géneros.
- **Periodo:** termina en 2018, así que no refleja películas ni hábitos posteriores.
- **Representatividad:** son usuarios de una plataforma en línea de películas, no una muestra de la población general ni de una institución dominicana. No se evaluó diferencia alguna entre grupos, porque el dataset no tiene variables demográficas.

**Usos prohibidos o no recomendados**
- Uso comercial sin permiso de GroupLens.
- Intentar reidentificar a los usuarios.
- Declarar o insinuar respaldo de la Universidad de Minnesota o de GroupLens.
- Afirmar conclusiones sobre grupos de personas (género, edad, país) o sobre la población general.
- Usar esta versión de desarrollo como base de resultados científicos compartidos, como advierte el propio README.

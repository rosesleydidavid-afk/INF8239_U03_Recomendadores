# INF8239_U03 — Ejercicio 05: Recomendador de películas (MovieLens)

Comparación reproducible de cuatro estrategias de recomendación (popularidad, contenido, factorización matricial e híbrido)
sobre MovieLens Latest Small, con corte temporal, métricas de ranking, cobertura, diversidad, arranque en frío y frontera de Pareto.

## Pregunta
¿Qué estrategia conviene para una plataforma de películas si importan los aciertos, la variedad de lo recomendado y el costo?

## Datos
- **Dataset:** MovieLens Latest Small (GroupLens, Universidad de Minnesota).
- **Redistribución:** **no se incluye en este repositorio** (la licencia lo prohíbe). Se descarga con `scripts/download_data.py`.
- **Procedencia, hash SHA-256 y fecha de descarga:** `reports/dataset_provenance.json`.
- **Ficha del dataset y reglas de uso:** `docs/DATASET_CARD.md`.
- 610 usuarios; 100,226 valoraciones de entrenamiento; 9,701 películas en el universo candidato.

## Estructura
```
scripts/   download_data.py · lab08_content.py · lab09_hybrid.py
src/inf8239_u03/   config.py · data.py · recommenders.py · evaluation.py · scorers.py
tests/     pruebas de métricas y puntuadores
notebooks/ 05_entrega_recomendadores.ipynb  (lee reports/, no reentrena)
reports/   resultados (CSV, JSON, figuras)
docs/      DATASET_CARD.md · SYSTEM_CARD.md
```

## Cómo reproducir
Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12.
```
uv sync
uv run python scripts/download_data.py      # descarga MovieLens y registra su procedencia
uv run python scripts/lab08_content.py      # popularidad suavizada y contenido por géneros
uv run python scripts/lab09_hybrid.py       # comparación completa (varios minutos: 9 modelos)
uv run pytest -q                            # esperado: 19 passed
```
Después abrir `notebooks/05_entrega_recomendadores.ipynb` y ejecutar todo (tarda segundos).
Opción rápida de prueba: `uv run python scripts/lab09_hybrid.py --quick`.

## Protocolo de evaluación
- **Corte temporal "leave-last-out":** la última valoración de cada usuario se reserva para evaluar.
- **Usuarios evaluables:** 587 de 610 (23 se excluyen porque su película reservada no aparece en el entrenamiento).
- **Métricas:** HitRate@10, NDCG@10, MRR, cobertura@10, diversidad de géneros@10, concentración en lo popular, RMSE (solo factorización), tiempo y memoria.
- **Semillas:** 42, 43, 44 · **Factores:** 10, 20, 40 · **Alphas del híbrido:** 0.25 y 0.75 · **Épocas:** 12.
- Regla de empates: a igual puntuación gana el ítem de menor índice.

## Resultados principales
| Método | HitRate@10 | NDCG@10 | Cobertura@10 | Concentración en populares |
|---|---|---|---|---|
| Popularidad (conteo) | 0.0443 | 0.0202 | 1.2 % | 100 % |
| Híbrido, 20 factores, alpha 0.75 | 0.0392 | 0.0184 | 7.6 % | 88 % |
| Factorización, 20 factores | 0.0386 | 0.0185 | 7.4 % | 88 % |
| Contenido (géneros) | 0.0017 | 0.0007 | 7.8 % | 22 % |
| Azar | 0.0006 | 0.0002 | 45 % | 10 % |

**Lectura honesta**
- Con 587 usuarios, un HitRate de 0.04 son unos 23–26 aciertos. Popularidad (26) e híbrido (23) difieren en unos 3 usuarios y sus intervalos del 95 % se solapan: **no se puede afirmar que uno sea mejor**.
- Más factores bajan el RMSE (1.030 → 1.025 → 1.019) pero no mejoran el ranking: el mejor HitRate de la factorización fue con 20 factores.
- **El híbrido no superó a la factorización sola** (0.0392 frente a 0.0386).
- Arranque en frío: 59 usuarios; el respaldo de popularidad acierta 2 (3.4 %). Muestra muy pequeña.
- En la frontera de Pareto (HitRate frente a tiempo de ajuste) solo queda la popularidad, porque no se entrena; no implica que sea la mejor opción de producto.

## Configuración elegida
**Híbrido con 20 factores y alpha 0.75, con popularidad como respaldo para usuarios nuevos.** Justificación: precisión indistinguible de la mejor,
unas 6 veces más cobertura que la popularidad, menor concentración en películas populares y la mitad de memoria que 40 factores.
Detalles, riesgos y límites en `docs/SYSTEM_CARD.md`.

## Limitaciones
- Un solo corte temporal; 94 usuarios tienen empate en la fecha de su última valoración, de modo que la película reservada entre empatadas es ambigua.
- Las valoraciones pasadas no miden satisfacción real.
- Cuadrícula pequeña y 12 épocas de SGD; el RMSE (≈1.02) podría mejorar con más ajuste.
- Los tiempos tienen ruido de la máquina; no se interpretan como una ley.

## Declaración de uso de IA
Esta declaración debe ser **verdadera**. Antes de entregar, revisar cada línea y editar lo que no corresponda.

**Herramienta:** Claude (Anthropic), usado como asistente paso a paso.

**Qué hizo la IA** (según el registro de la sesión)
- Redactó o amplió `scripts/download_data.py`, `lab08_content.py`, `lab09_hybrid.py`, `src/inf8239_u03/evaluation.py`, `src/inf8239_u03/scorers.py`, las pruebas `tests/test_evaluation.py` y `tests/test_scorers.py`, el notebook, `docs/DATASET_CARD.md`, `docs/SYSTEM_CARD.md` y este README.
- Probó el código en su propio entorno con **datos sintéticos**; nunca vio los datos reales de MovieLens. Los números reales los produjo la estudiante en su computadora.
- Propuso la interpretación de resultados y la configuración elegida.

**Qué hice yo** (marcar solo lo que se hizo de verdad)
- [ ] Ejecuté todos los comandos en mi máquina y copié sus salidas reales.
- [ ] Descargué MovieLens y verifiqué `reports/dataset_provenance.json`.
- [ ] Ejecuté `pytest` (19 passed) y el notebook completo.
- [ ] Abrí y revisé las figuras y tablas generadas.
- [ ] Leí el código de las métricas (`evaluation.py`) y puedo explicar HitRate@K, NDCG@K y MRR.
- [ ] Puedo explicar por qué el RMSE y el ranking no coinciden.
- [ ] Comprobé que la declaración y los números del README coinciden con `reports/`.

**Qué no entiendo todavía o no verifiqué:** COMPLETAR (escribir con honestidad).

## Licencia y atribución
El código de este repositorio es de uso académico. MovieLens: F. Maxwell Harper y Joseph A. Konstan, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4), 2015. Sujeto a la licencia de GroupLens.

## Declaración de uso de IA (continuación): prompts, verificaciones y correcciones

**Prompts relevantes** (mensajes míos, resumidos; la sesión completa se conserva)
- Pegué el mandato y la rúbrica del Ejercicio 05 y los manuales LAB08 y LAB09, y pedí avanzar paso a paso, mostrando un checklist, como si yo no supiera nada.
- Pegué la salida real de cada comando que ejecuté (descarga y auditoría, LAB08, corrida rápida y completa de LAB09, pruebas, Git) para que se interpretara.
- Pedí revisar el proyecto base contra la rúbrica antes de modificarlo.

**Correcciones hechas durante el trabajo** (errores detectados y arreglados)
- El proyecto base evaluaba solo el híbrido: se añadió la comparación de popularidad, contenido, factorización, híbrido y azar con el mismo corte y universo.
- El script no tenía semillas ni arranque en frío: se añadieron tres semillas y usuarios retirados del entrenamiento.
- La fecha, la URL y el hash de la descarga solo se imprimían: ahora se guardan en `reports/dataset_provenance.json`.
- El `.gitignore` base excluía `reports/`, que es la evidencia: se corrigió; `data/raw/` sigue excluido.
- La comparación de géneros en LAB08 se hacía como texto y se cambió a conjuntos.
- El azar entraba en la frontera de Pareto: se excluyó por ser un piso de referencia.
- El perfil de usuarios se mostraba solo para un alpha y con un modelo arbitrario: ahora se muestra para ambos alphas con el modelo de 20 factores.

**Qué verificó la IA y qué verifiqué yo**
- La IA probó el código con datos sintéticos (pruebas y corridas rápidas); no tuvo acceso a MovieLens real.
- Yo ejecuté el código con los datos reales en mi computadora, obtuve `19 passed` y la corrida completa, y generé el notebook con los resultados guardados.
- Los números del README, de la System Card y del PDF se copiaron de `reports/` y de las salidas reales que pegué.

**Qué no se inventó:** no hay referencias ni resultados fabricados; la única fuente externa es MovieLens (GroupLens), citada arriba.

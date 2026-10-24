# System Card — Recomendador de películas (Ejercicio 05, INF8239 U03)

## 1. Propósito y uso previsto
Sugerir un Top-10 de películas a cada usuario de una plataforma, a partir de su historial de valoraciones.
**Uso previsto:** ejercicio académico de comparación de estrategias de recomendación.
**No apto para:** producción comercial sin una validación adicional, decisiones sobre personas, ni recomendar contenido a menores sin controles propios.

## 2. Usuarios y decisión que apoya
Personas que buscan qué ver. Decisión: qué 10 títulos mostrar primero. El error de mostrar siempre lo mismo reduce el descubrimiento; el de mostrar títulos poco relacionados reduce la utilidad.

## 3. Datos
MovieLens Latest Small (GroupLens): 610 usuarios, valoraciones de 0.5 a 5, géneros por película. No se redistribuye; procedencia y hash en `reports/dataset_provenance.json` y reglas de uso en `docs/DATASET_CARD.md`.
No contiene datos demográficos, por lo que **no se pudo evaluar equidad entre grupos de usuarios**.

## 4. Sistema
| Componente | Descripción |
|---|---|
| Popularidad | Conteo de valoraciones (y media suavizada tipo bayesiano) |
| Contenido | Perfil del usuario sobre géneros (TF-IDF) y similitud coseno |
| Factorización | SGD, 12 épocas, 10/20/40 factores |
| Híbrido | Top-100 de la factorización reordenado con `alpha·colaborativo + (1−alpha)·contenido` (alpha = peso colaborativo) |
| Respaldo | Popularidad para usuarios sin historial (se rotula como no personalizada) |

**Configuración elegida:** híbrido, 20 factores, alpha 0.75, con respaldo de popularidad.

## 5. Evaluación
Corte temporal leave-last-out; 587 usuarios evaluables; universo de 9,701 películas; K = 10; 3 semillas.

| Método | HitRate@10 | NDCG@10 | Cobertura | Diversidad géneros | Concentración populares |
|---|---|---|---|---|---|
| Popularidad | 0.0443 | 0.0202 | 1.2 % | 0.79 | 100 % |
| Híbrido f20 a0.75 | 0.0392 | 0.0184 | 7.6 % | 0.76 | 88 % |
| Factorización f20 | 0.0386 | 0.0185 | 7.4 % | 0.80 | 88 % |
| Híbrido f20 a0.25 | 0.0284 | 0.0122 | 10.4 % | 0.54 | 82 % |
| Contenido | 0.0017 | 0.0007 | 7.8 % | 0.18 | 22 % |
| Azar | 0.0006 | 0.0002 | 45 % | 0.83 | 10 % |

**Incertidumbre:** con 587 usuarios, el margen aproximado del 95 % del HitRate es de ±0.013 a ±0.017. Las diferencias entre popularidad, híbrido y factorización caen dentro de ese margen.
**Arranque en frío:** 59 usuarios retirados del entrenamiento; el respaldo de popularidad acierta 2 (3.4 %). Contenido con 5 valoraciones y azar: 0 aciertos.

## 6. Riesgos y mitigaciones
| Riesgo | Evidencia aquí | Mitigación propuesta |
|---|---|---|
| Sesgo de popularidad | La popularidad concentra el 100 % de sus listas en títulos populares y cubre 1.2 % del catálogo | Reordenar con contenido; monitorear cobertura y concentración |
| Burbuja de filtro | Con alpha 0.25 la diversidad de géneros cae de 0.76 a 0.53 | Preferir alpha alto y vigilar la diversidad |
| Bucle de retroalimentación | Lo recomendado influye en lo que luego se valora | Reservar una parte de exploración; reentrenar con cuidado |
| Usuarios nuevos | La factorización no puede puntuarlos | Respaldo de popularidad, rotulado como no personalizado |
| Privacidad | Historiales de valoraciones por usuario | Usar solo datos autorizados; no combinar con otras fuentes |
| Equidad | No hay datos demográficos | No se puede afirmar equidad; requeriría datos y consentimiento |

## 7. Limitaciones conocidas
- Resultados de un solo corte temporal; 94 usuarios tienen empate en la fecha de la última valoración (la película reservada es ambigua entre las empatadas).
- Las valoraciones pasadas no miden satisfacción real: una película no valorada puede gustar.
- El híbrido **no superó** a la factorización sola ni a la popularidad en precisión.
- Muestra de arranque en frío muy pequeña (59 usuarios, 2 aciertos).
- Cuadrícula pequeña y entrenamiento corto; RMSE ≈ 1.02.
- Tiempos con ruido de la máquina.

## 8. Monitoreo sugerido
Seguir HitRate y NDCG, cobertura, concentración en populares y diversidad por semana; revisar cuando la cobertura baje o la concentración suba.

## 9. Trazabilidad
Código en `scripts/` y `src/inf8239_u03/`; resultados en `reports/` (`lab09_audit.json` guarda argumentos y entorno); pruebas con `uv run pytest -q`.
Uso de IA: ver la declaración en `README.md`.

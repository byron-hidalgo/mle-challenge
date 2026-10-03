# Challenge MLE – Byron Hidalgo

API desplegada: https://mle-api-iik5n55rmq-uc.a.run.app

## I. Modelo
- Modelo elegido: **XGBoost con balanceo de clases** (`scale_pos_weight = n_puntuales / n_atrasados`),
  10 features más importantes (notebook, sección 6)
- XGBoost balanceado y Regresión Logística balanceada empatan (recall atrasos ≈ 0.69). Se elige XGBoost
  porque el template ya lo anticipa. Sin balanceo el recall de atrasos cae a ≈ 0.01, lo que lo hace inútil.
- Métricas (test 33 %, random_state=42): recall atrasos 0.688, F1 atrasos 0.366, accuracy 0.553.
- Supuestos: atraso = mantenemos qeu sea con más de 15 min como indica el notebook, categorías no vistas quedan en 0 y umbral 0.5.
  Si no hay modelo entrenado, `predict` retorna 0 con un warning.

### Bugs encontrados en el notebook 
| Función | Problema |
|---|---|
| `is_high_season` | compara fechas con hora, clasifica mal 730 filas |
| `get_rate_from_column` | calcula total/atrasos (invertido), por ejemplo internacional 4.43 en vez de 22.6 % |
| varios | args de barplot, `training_data` sin usar|

### Cambio en tests
`tests/pipelines/test_training.py` usaba `FEATURES_COLS` sin importarlo (NameError). Se agregó una sola
línea: `from challenge.model import FEATURES_COLS`. Los tests de model no se tocaron.

## II–V. Arquitectura (GCP, us-central1)
- **GCS**: `<proyecto>-mle-input` (CSV de entrada) y `<proyecto>-mle-artifacts` (modelos + `latest.txt` + métricas).
- **BigQuery** `mle_challenge`: `raw_flights` (entrenamiento), `serving_input`, `predictions`.
- **Cloud Run Jobs**: `delay-training` (BigQuery → modelo en GCS) y `delay-serving` (modelo + CSV → `predictions`).
- **Cloud Run Service** `mle-api` (FastAPI): `/health`, `POST /pipeline/train`, `POST /pipeline/predict`,
  `GET /pipeline/predict/results` (paginado y filtros `opera`, `tipovuelo`, `mes`).
- Decisiones: endpoints `async` del template + `run_in_threadpool` para no bloquear; lock para no
  entrenar dos veces a la vez (409); caché de resultados de 60 s; la API usa siempre el último modelo (`latest.txt`).

## VI. Infraestructura y CI/CD
- `infra/bootstrap.sh`: crea todo por CLI, sin consola web (APIs, cuenta de servicio y roles, buckets,
  dataset y tablas, Artifact Registry, imágenes, jobs y API). Es idempotente.
```bash
  PROJECT_ID=<tu-proyecto> bash infra/bootstrap.sh
```
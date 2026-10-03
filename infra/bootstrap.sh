#!/usr/bin/env bash
# =============================================================================
# Crea desde cero toda la infraestructura del challenge en un proyecto de GCP.
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PROJECT_ID="${PROJECT_ID:?Debes definir PROJECT_ID, ej: PROJECT_ID=mi-proyecto bash infra/bootstrap.sh}"
REGION="${REGION:-us-central1}"

SA_NAME="mle-challenge-sa"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

INPUT_BUCKET="${PROJECT_ID}-mle-input"
ARTIFACTS_BUCKET="${PROJECT_ID}-mle-artifacts"
BQ_DATASET="mle_challenge"
AR_REPO="mle-challenge"

IMAGE_PREFIX="${REGION}-docker.pkg.dev/${PROJECT_ID}/${AR_REPO}"
API_SERVICE="mle-api"
TRAINING_JOB="delay-training"
SERVING_JOB="delay-serving"

# Variables de entorno que usan la API y los jobs en GCP.
ENV_VARS="DEPLOYMENT_MODE=gcp,GCP_PROJECT_ID=${PROJECT_ID},GCP_REGION=${REGION}"
ENV_VARS="${ENV_VARS},GCP_BUCKET_ARTIFACTS=${ARTIFACTS_BUCKET},GCP_BUCKET_INPUT=${INPUT_BUCKET}"
ENV_VARS="${ENV_VARS},BQ_DATASET=${BQ_DATASET}"

gcloud config set project "${PROJECT_ID}"

# ---------------------------------------------------------------------------
# 1. Habilitar las APIs de GCP que usa el proyecto
# ---------------------------------------------------------------------------
echo ">> 1/7 Habilitando APIs"
gcloud services enable \
  run.googleapis.com \
  bigquery.googleapis.com \
  storage.googleapis.com \
  artifactregistry.googleapis.com \
  iam.googleapis.com \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com

# ---------------------------------------------------------------------------
# 2. Cuenta de servicio única
# ---------------------------------------------------------------------------
echo ">> 2/7 Cuenta de servicio"
if ! gcloud iam service-accounts describe "${SA_EMAIL}" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SA_NAME}" --display-name="MLE challenge service account"
fi

# Permisos mínimos para leer/escribir datos, modelos e imágenes, y lanzar los jobs.
for ROLE in \
  roles/bigquery.dataEditor \
  roles/bigquery.jobUser \
  roles/storage.objectAdmin \
  roles/run.developer \
  roles/iam.serviceAccountUser \
  roles/artifactregistry.writer \
  roles/logging.logWriter; do
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" --role="${ROLE}" --condition=None --quiet >/dev/null
done

# ---------------------------------------------------------------------------
# 3. Buckets de Cloud Storage y carga de los CSV
# ---------------------------------------------------------------------------
echo ">> 3/7 Buckets de Cloud Storage"
for BUCKET in "${INPUT_BUCKET}" "${ARTIFACTS_BUCKET}"; do
  if ! gcloud storage buckets describe "gs://${BUCKET}" >/dev/null 2>&1; then
    gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" --uniform-bucket-level-access
  fi
done

gcloud storage cp data/data.csv "gs://${INPUT_BUCKET}/data.csv"
gcloud storage cp data/serving_input.csv "gs://${INPUT_BUCKET}/serving_input.csv"

# ---------------------------------------------------------------------------
# 4. BigQuery: dataset y tablas cargadas desde los CSV del bucket
# ---------------------------------------------------------------------------
echo ">> 4/7 BigQuery"
bq --location="${REGION}" mk --force --dataset "${PROJECT_ID}:${BQ_DATASET}"

# Esquema explícito con los mismos nombres de columna del CSV.
RAW_SCHEMA="Fecha-I:STRING,Vlo-I:STRING,Ori-I:STRING,Des-I:STRING,Emp-I:STRING"
RAW_SCHEMA="${RAW_SCHEMA},Fecha-O:STRING,Vlo-O:STRING,Ori-O:STRING,Des-O:STRING,Emp-O:STRING"
RAW_SCHEMA="${RAW_SCHEMA},DIA:INTEGER,MES:INTEGER,AÑO:INTEGER,DIANOM:STRING,TIPOVUELO:STRING"
RAW_SCHEMA="${RAW_SCHEMA},OPERA:STRING,SIGLAORI:STRING,SIGLADES:STRING"

bq --location="${REGION}" load --replace --source_format=CSV --skip_leading_rows=1 \
  "${PROJECT_ID}:${BQ_DATASET}.raw_flights" "gs://${INPUT_BUCKET}/data.csv" "${RAW_SCHEMA}"

bq --location="${REGION}" load --replace --source_format=CSV --skip_leading_rows=1 \
  "${PROJECT_ID}:${BQ_DATASET}.serving_input" "gs://${INPUT_BUCKET}/serving_input.csv" \
  "OPERA:STRING,TIPOVUELO:STRING,MES:INTEGER"

# ---------------------------------------------------------------------------
# 5. Artifact Registry: repositorio para las imágenes Docker
# ---------------------------------------------------------------------------
echo ">> 5/7 Artifact Registry"
if ! gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" >/dev/null 2>&1; then
  gcloud artifacts repositories create "${AR_REPO}" --repository-format=docker --location="${REGION}"
fi

# ---------------------------------------------------------------------------
# 6. Construir y subir las 3 imágenes
# ---------------------------------------------------------------------------
echo ">> 6/7 Imágenes Docker"
gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
for NAME in api training serving; do
  docker build -f "Dockerfile.${NAME}" -t "${IMAGE_PREFIX}/${NAME}:latest" .
  docker push "${IMAGE_PREFIX}/${NAME}:latest"
done

# ---------------------------------------------------------------------------
# 7. Desplegar los 2 Cloud Run Jobs y la API
# ---------------------------------------------------------------------------
echo ">> 7/7 Cloud Run"
gcloud run jobs deploy "${TRAINING_JOB}" \
  --image="${IMAGE_PREFIX}/training:latest" --region="${REGION}" \
  --service-account="${SA_EMAIL}" --set-env-vars="${ENV_VARS}" \
  --memory=2Gi --cpu=1 --task-timeout=1800 --max-retries=0

gcloud run jobs deploy "${SERVING_JOB}" \
  --image="${IMAGE_PREFIX}/serving:latest" --region="${REGION}" \
  --service-account="${SA_EMAIL}" --set-env-vars="${ENV_VARS}" \
  --memory=1Gi --cpu=1 --task-timeout=1800 --max-retries=0

# La API espera a los jobs, por eso su timeout es de 1 hora (el máximo de Cloud Run).
gcloud run deploy "${API_SERVICE}" \
  --image="${IMAGE_PREFIX}/api:latest" --region="${REGION}" \
  --service-account="${SA_EMAIL}" \
  --set-env-vars="${ENV_VARS},TRAINING_JOB_NAME=${TRAINING_JOB},SERVING_JOB_NAME=${SERVING_JOB}" \
  --memory=1Gi --cpu=1 --timeout=3600 --min-instances=0 --max-instances=3 \
  --allow-unauthenticated

API_URL="$(gcloud run services describe "${API_SERVICE}" --region="${REGION}" --format='value(status.url)')"
echo ""
echo "Listo. URL de la API: ${API_URL}"
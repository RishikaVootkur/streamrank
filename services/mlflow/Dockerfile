# MLflow tracking server with the Postgres driver, running as a non-root user.
FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.13 \
    && useradd --create-home --uid 10001 mlflow \
    && mkdir -p /mlartifacts && chown mlflow /mlartifacts
USER mlflow
EXPOSE 5000
CMD ["sh", "-c", "mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --artifacts-destination /mlartifacts --serve-artifacts --allowed-hosts '*'"]

FROM python:3.9-slim
WORKDIR /app

# Copy requirements and install them first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application script and the model artifacts directory explicitly.
COPY ./app/main.py .
COPY ./app/model_artifacts ./model_artifacts

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
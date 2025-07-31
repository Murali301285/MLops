FROM python:3.9-slim
WORKDIR /app

# Copy requirements and install them first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the mlruns directory
COPY ./app ./app
COPY ./mlruns ./mlruns

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
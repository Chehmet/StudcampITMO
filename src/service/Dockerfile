FROM python:3.11-slim
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем зависимости
COPY requirements.txt ./
RUN pip install --no-cache-dir --timeout 1000 --retries 3 -i https://pypi.org/simple/ -r requirements.txt

# Копируем код приложения
COPY . .

# Команда запуска
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]

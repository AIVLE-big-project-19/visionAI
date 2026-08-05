FROM python:3.11-slim
WORKDIR /app

# opencv-python-headless가 런타임에 필요로 하는 공유 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY data data
COPY bestv2.pt .

ENV MODEL_PATH=/app/bestv2.pt

EXPOSE 8001
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]

FROM  python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY fast_api_app/ /app/fast_api_app/

COPY fast_api_app/requirements.txt .

COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 8000 

CMD ["uvicorn", "fast_api_app.main:app", "--host", "0.0.0.0", "--port", "8000"]


# 1 
FROM python:3.7

# 2
RUN apt-get update && apt-get install -y \
  build-essential \
  gfortran \
  git \
  wget \
  liblapack-dev \
  libopenblas-dev

RUN pip install Flask pyopenssl flask_cors gunicorn pandas numpy gcsfs flask_caching firebase_admin flask_restful fastprogress matplotlib scikit-learn numba sentry-sdk[flask]

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 443

# # 5
CMD exec gunicorn --certfile=../config/fullchain.pem --keyfile=../config/privkey.pem --bind :$PORT --workers 1 --timeout 200 --threads 4 app:app
FROM python:3.12
ARG APP_VERSION
ENV APP_VERSION=$APP_VERSION
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y sqlite3 libsqlite3-dev
ENV PORT 8501
EXPOSE 8501
CMD sh -c 'streamlit run --server.port $PORT medDocSearch.py'
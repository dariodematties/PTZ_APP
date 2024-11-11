FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /app
#COPY requirements.txt requirements.txt
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV TRANSFORMERS_CACHE=/hf_cache/

ENTRYPOINT ["python", "./main.py"]

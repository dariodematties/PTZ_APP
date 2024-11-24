FROM nvcr.io/nvidia/pytorch:24.06-py3

WORKDIR /app
#COPY requirements.txt requirements.txt
COPY . ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
ENV TRANSFORMERS_CACHE=/hf_cache/

# Download the model during the build process
RUN python -c "\
import torch; \
from transformers import AutoProcessor, AutoModelForCausalLM; \
model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', torch_dtype=torch.float32, trust_remote_code=True); \
processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"

ENTRYPOINT ["python", "./main.py"]

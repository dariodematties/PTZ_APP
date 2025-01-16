FROM nvcr.io/nvidia/pytorch:24.06-py3

# 1. Copy code
WORKDIR /app
COPY . ./

# 2. Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 3. Create a local cache directory for the model
RUN mkdir -p /hf_cache/microsoft/Florence-2-large

# 4. Download the model files info /hf_cache/microsoft/Florence-2-large
RUN huggingface-cli download \
	microsoft/Florence-2-large \
	--repo-type model \
	--cache-dir /hf_cache \
	--local-dir /hf_cache/microsoft/Florence-2-large \
	--resume \
	--force  # <-- ensures we overwrite any existing files or resume a download

# 5. Set the environment variables for offline mode
ENV HF_HOME=/hf_cache
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1




# ENV TRANSFORMERS_CACHE=/hf_cache/

# # Download the model during the build process
# RUN python -c "\
# import torch; \
# from transformers import AutoProcessor, AutoModelForCausalLM; \
# model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', torch_dtype=torch.float32, trust_remote_code=True); \
# processor = AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)"
#
ENTRYPOINT ["python", "./main.py"]

import torch
from transformers import AutoProcessor, AutoModelForCausalLM 
from source.utils import get_rewards_from_image_and_object



def get_label_from_image_and_object(image, prompt, model, processor):
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    device = "cpu"
    torch_dtype = torch.float32


    return get_rewards_from_image_and_object(processor, image, prompt, device, torch_dtype, model)


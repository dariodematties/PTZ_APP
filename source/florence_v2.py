import torch
from transformers import AutoProcessor, AutoModelForCausalLM 
from source.utils import get_rewards_from_image_and_object



def get_label_from_image_and_object(image, prompt):
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    device = "cpu"
    torch_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    #url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
    #image = Image.open(requests.get(url, stream=True).raw)

    #prompt = "<MORE_DETAILED_CAPTION>"
    #answer = run_example(processor, image, device, torch_dtype, model, prompt)['<MORE_DETAILED_CAPTION>']
    #print(answer)

    #task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    #results = run_example(processor, image, device, torch_dtype, model, task_prompt, text_input=answer)
    #print(results)

    return get_rewards_from_image_and_object(processor, image, prompt, device, torch_dtype, model)

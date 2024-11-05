import os
import json
import numpy as np
from pathlib import Path

def run_example(processor, image, device, torch_dtype, model, task_prompt, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    return parsed_answer




def bbox_area(bbox):
    x1=bbox[0]
    y1=bbox[1]
    x2=bbox[2]
    y2=bbox[3]
    assert x1 < x2
    assert y1 < y2
    x_len = x2-x1
    y_len = y2-y1
    return x_len*y_len




def get_rewards_from_bboxes(bboxes, labels, image):
    ratios = []
    for bbox, label in zip(bboxes, labels):
        #print('bbox: ', bbox)
        #print('label: ', label)
        #print('ratio: ', bbox_area(bbox)/np.prod(image.size))
        ratios.append(bbox_area(bbox)/np.prod(image.size))

    #return sum(ratios)/len(ratios)
    return ratios, bboxes, labels





def get_rewards_from_image_and_object(processor, image, prompt, device, torch_dtype, model, text_input=None):
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
    results = run_example(processor, image, device, torch_dtype, model, task_prompt, text_input=prompt)
    bboxes=results['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    labels=results['<CAPTION_TO_PHRASE_GROUNDING>']['labels']

    return get_rewards_from_bboxes(bboxes, labels, image)


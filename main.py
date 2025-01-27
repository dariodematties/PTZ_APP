# Defining main function 
import os
import yaml
import argparse

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
from source.bring_data import center_and_maximize_object, get_image_from_ptz_position, publish_images


def get_argparser():
    parser = argparse.ArgumentParser("PTZ APP")
    # PTZ application
    parser.add_argument(
        "-ki",
        "--keepimages",
        action="store_true",
        help="Keep collected images in persistent folder for later use",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        help="An integer with the number of iterations (PTZ rounds) to be run (default=5).",
        type=int,
        default=5,
    )
    parser.add_argument(
        "-obj",
        "--object",
        help="The object to capture with the camera.",
        type=str,
        default="animal",
    )
    parser.add_argument(
        "-un",
        "--username",
        help="The username of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-pw",
        "--password",
        help="The password of the PTZ camera.",
        type=str,
        default="",
    )
    parser.add_argument(
        "-ip", "--cameraip", help="The ip of the PTZ camera.", type=str, default=""
    )
    parser.add_argument(
        "-ps", "--panstep", help="The step of pan in degrees.", type=int, default=15
    )
    parser.add_argument(
        "-tv", "--tilt", help="The tilt value in degrees.", type=int, default=0
    )
    parser.add_argument(
        "-zm", "--zoom", help="The zoom value.", type=int, default=1
    )
    parser.add_argument(
        "-mod", "--model", help="The model to use.", type=str, default="Florence-2-base"
    )

    return parser

















def look_for_object(args):
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cpu"
    torch_dtype = torch.float32

    object_ = args.object
    pans = [angle for angle in range(0, 360, args.panstep)]
    tilts = [args.tilt for _ in range(len(pans))]
    zooms = [args.zoom for _ in range(len(pans))]

    if args.model == "Florence-2-base":
        model_dir = "/hf_cache/microsoft/Florence-2-base"
    elif args.model == "Florence-2-large":
        model_dir = "/hf_cache/microsoft/Florence-2-large"
    else:
        raise ValueError("Model can only be Florence-2-base or Florence-2-large but got: ", args.model)

    model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            local_files_only=True  # <--- prevents huggingface from hitting the internet
            ).to(device)
    model.eval()

    processor = AutoProcessor.from_pretrained(
            model_dir,
            trust_remote_code=True,
            local_files_only=True  # <--- prevents huggingface from hitting the internet
            )

    
    for iteration in range(args.iterations):
        for pan, tilt, zoom in zip(pans, tilts, zooms):
            policy_net=None
            image_path, LABEL = get_image_from_ptz_position(args, object_, pan, tilt, zoom, model, processor)
            reward = LABEL['reward']
            print(image_path)
            if LABEL is None or reward > 0.99:
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print(f'             no object found                ')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
            else:
                # Chooses the label to follow
                label = LABEL['label']
                bbox = LABEL['bbox']
                reward = LABEL['reward']
                print('reward: ', reward)
                print('type(reward): ', type(reward))
                image = Image.open(image_path)
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print(f'         following {label} object           ')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                print('<<<<<<>>>>>>---------------------<<<<<<>>>>>>')
                center_and_maximize_object(args, bbox, image, reward)

            os.remove(image_path)

        publish_images()






def main():
    args = get_argparser().parse_args()
    look_for_object(args)





# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

# Defining main function 
import os
import yaml
import argparse

import torch
from PIL import Image
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

    return parser

















def look_for_object(args):
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    #torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cpu"
    torch_dtype = torch.float32

    #object_ = 'robot object'
    #object_ = 'very tiny and distant objects'
    #object_ = 'a camera dome'
    object_ = args.object
    #pans = [150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150, 150]
    pans = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
    tilts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    zooms = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #zooms = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    #zooms = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    #zooms = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    #zooms = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]


    pans = [0, 90, 180, 270, 345]
    tilts = [0, 0, 0, 0, 0]
    zooms = [1, 1, 1, 1, 1]

    
    for iteration in range(args.iterations):
        for pan, tilt, zoom in zip(pans, tilts, zooms):
            policy_net=None
            image_path, LABEL = get_image_from_ptz_position(args, object_, pan, tilt, zoom)
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
                os.remove(image_path)
                center_and_maximize_object(args, bbox, image, reward)

        publish_images()






def main():
    args = get_argparser().parse_args()
    look_for_object(args)





# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

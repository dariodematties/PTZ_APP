import os
import shutil
import logging
import math
import random
import time
import datetime
from PIL import Image
from pathlib import Path
from source import sunapi_control as camera_control
from source.florence_v2 import get_label_from_image_and_object


logger = logging.getLogger(__name__)



try:
    # ! Note this assumes the code is running in a container
    tmp_dir = Path("/imgs")
    tmp_dir.mkdir(exist_ok=True, mode=0o777)
except OSError:
    logger.warning(
        "Could not create directories, will use default paths and the code might break"
    )







def center_and_maximize_object(args, bbox, image, reward=None):
    x1, y1, x2, y2 = bbox
    image_width, image_height = image.size
    
    print(f'x1: {x1}')
    print(f'y1: {y1}')
    print(f'x2: {x2}')
    print(f'y2: {y2}')
    print(f'image_width: {image_width}')
    print(f'image_height: {image_height}')
    # Calculate the center of the bounding box
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2
    
    # Calculate the center of the image
    image_center_x = image_width / 2
    image_center_y = image_height / 2
    
    # Calculate the difference between the centers in pixels
    diff_x = image_center_x - bbox_center_x
    diff_y = image_center_y - bbox_center_y

    if diff_x < 0:
        print('MOVE RIGHT')
    else:
        print('MOVE LEFT')
    
    if diff_y < 0:
        print('MOVE DOWN')
    else:
        print('MOVE UP')
    
    try:
        Camera1 = camera_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except Exception as e:
        logger.error("Error when getting camera: %s", e)



    _, _, zoom_level = Camera1.requesting_cameras_position_information()
    print(f'zoom_level: {zoom_level}')

    # Get current FOV based on zoom level
    current_h_fov, current_v_fov = get_fov_from_zoom(zoom_level)
    #current_h_fov, current_v_fov = get_current_fov(zoom_level)
    print('current_h_fov: ', current_h_fov)
    print('current_v_fov: ', current_v_fov)
    
    # Convert pixel difference to degrees
    pan = -(diff_x / image_width) * current_h_fov
    tilt = -(diff_y / image_height) * current_v_fov




    # Move the camera to center the object
    print('Move the camera to center the object')
    print(f'Pan: {pan}')
    print(f'Tilt: {tilt}')
    try:
        Camera1.relative_control(pan=pan, tilt=tilt)#, zoom=zoom)
    except Exception as e:
        logger.error("Error when setting relative position: %s", e)

    time.sleep(3)
    ## Move the camera to center the object
    #camera.relative_control(pan=pan, tilt=tilt, zoom=0)
    
    # Calculate the current size of the bounding box
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Calculate the zoom factor to maximize the object size
    zoom_factor_x = image_width / bbox_width
    zoom_factor_y = image_height / bbox_height
    zoom_factor = min(zoom_factor_x, zoom_factor_y)
    print('zoom_factor_x: ', zoom_factor_x)
    print('zoom_factor_y: ', zoom_factor_y)
    print('zoom_factor: ', zoom_factor)
    
    # Calculate relative zoom (assuming current zoom_level range is 0 to 1)
    #relative_zoom = (zoom_factor - 1) * (1 - zoom_level)

    # Calculate relative zoom
    mz=1
    MZ=40
    current_zoom_factor = zoom_level / MZ
    #current_zoom_factor = (zoom_level - mz) / (MZ - mz)
    target_zoom_factor = current_zoom_factor * zoom_factor
    relative_zoom = target_zoom_factor * (MZ - mz) - zoom_level
    #relative_zoom = (target_zoom_factor * (MZ - mz) + mz) - zoom_level
    print('current_zoom_factor: ', current_zoom_factor)
    print('target_zoom_factor: ', target_zoom_factor)
    
    # Apply zoom (ensuring we don't exceed the maximum zoom)
    print('Apply zoom (ensuring we don\'t exceed the maximum zoom)')
    print(f'Relative zoom: {relative_zoom}')
    try:
        Camera1.relative_control(pan=0, tilt=0, zoom=relative_zoom)#, zoom=zoom)
    except Exception as e:
        logger.error("Error when setting relative position: %s", e)

    time.sleep(3)
    ## Apply zoom (ensuring we don't exceed the maximum zoom)
    #camera.relative_control(pan=0, tilt=0, zoom=relative_zoom)
    if reward is not None and reward < 0.99:
        image_path = grab_image(camera=Camera1, args=args, action=0)
        #collect_images(True)
        #print('-------------------------')
        #print('-------------------------')
        #print('-------------------------')
        #print('image_path: ', image_path)
        #print('image_path: ', image_path)
        #print('-------------------------')
        #print('-------------------------')
        #print('-------------------------')
        #Plugin.upload_file(image_path, timestamp=datetime.datetime.now())
        #Plugin.upload_file(path=image_path, meta='camera_dome', timestamp=datetime.datetime.now())
        #os.remove(image_path)


def get_image_from_ptz_position(args, object_, pan, tilt, zoom):
    iterations = args.iterations

    try:
        Camera1 = camera_control.CameraControl(
            args.cameraip, args.username, args.password
        )
    except Exception as e:
        logger.error("Error when getting camera: %s", e)

    # reset the camera to its original position
    Camera1.absolute_control(pan, tilt, zoom)
    time.sleep(1)

    tmp_dir.mkdir(exist_ok=True, mode=0o777)

    aux_image_path = grab_image(camera=Camera1, args=args, action=0)
    image = Image.open(aux_image_path)
    os.remove(aux_image_path)
    rewards, bboxes, labels = get_label_from_image_and_object(image, object_)
    print('rewards: ', rewards)
    if len(bboxes) > 0:
        #index = random.randint(0,len( labels )-1)
        index = rewards.index(min(rewards))
        print('index: ', index)
        LABEL = {'bbox': bboxes[index], 'label': labels[index], 'reward': rewards[index], 'first': True}
        bbox = LABEL['bbox']
    else:
        LABEL = None

    image_path = grab_image(camera=Camera1, args=args, action=random.randint(0,20))

    return image_path, LABEL


def publish_images():
    # run tar -cvf images.tar /imgs
    tar_images("images.tar", str(tmp_dir))
    # files = glob.glob("/imgs/*.jpg", recursive=True)
    shutil.rmtree(tmp_dir, ignore_errors=True)

    with Plugin() as plugin:
        ct = str(datetime.datetime.now())
        os.rename("images.tar", ct + "_images.tar")
        plugin.upload_file(ct + "_images.tar")


def get_fov_from_zoom(zoom_level):
    # Camera specifications
    min_focal_length = 4.25  # mm
    max_focal_length = 170  # mm
    h_wide, h_tele = 65.66, 1.88  # degrees
    v_wide, v_tele = 39.40, 1.09  # degrees
    min_zoom, max_zoom = 1, 40  # optical zoom range

    # Ensure zoom_level is within the valid range
    zoom_level = max(min_zoom, min(max_zoom, zoom_level))

    # Calculate current focal length based on zoom level
    focal_length = min_focal_length * zoom_level

    # Calculate the sensor dimensions
    sensor_width = 2 * min_focal_length * math.tan(math.radians(h_wide / 2))
    sensor_height = 2 * min_focal_length * math.tan(math.radians(v_wide / 2))

    # Calculate current FOV
    current_h_fov = math.degrees(2 * math.atan(sensor_width / (2 * focal_length)))
    current_v_fov = math.degrees(2 * math.atan(sensor_height / (2 * focal_length)))

    return current_h_fov, current_v_fov


def grab_image(camera, args, action):
    position = camera.requesting_cameras_position_information()

    pos_str = ",".join([str(p) for p in position])
    action_str = str(action)
    # ct stores current time
    ct = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    img_path = str(tmp_dir / f"{pos_str}_{action_str}_{ct}.jpg")
    #print('img_path: ', img_path)
    try:
        camera.snap_shot(img_path)
    # TODO: need to check what kind of exception is raised
    except Exception as e:
        logger.error("Error when taking snap shot: %s : %s", img_path, e)
        #if args.publish_msgs:
            #with Plugin() as plugin:
                #plugin.publish(
                    #"cannot.capture.image.from.camera", str(datetime.datetime.now())
                #)
        return None
    return img_path

def tar_images(output_filename, folder_to_archive):
    try:
        cmd = ["tar", "cvf", output_filename, folder_to_archive]
        output = subprocess.check_output(cmd).decode("utf-8").strip()
        logger.info(output)
    except Exception:
        logger.exception("Error when tar images")


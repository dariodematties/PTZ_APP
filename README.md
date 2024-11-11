# PTZ APP

This is an application for sending images of specific objects authonomously using PTZ cameras

## Build the container

`sudo docker buildx build --platform=linux/amd64,linux/arm64/v8 -t your_docker_hub_user_name/ptzapp -f Dockerfile --push .`

Then pull the container from dockerhub in the node

`sudo docker image pull your_docker_hub_user_name/ptzapp`

## Run the container on a dell blade

`sudo docker run --gpus all -it --rm your_docker_hub_user_name/ptzapp:latest -ki -it number_of_iterations -un camera_user_name -pw camera_password -ip camera_ip_address -obj object_name`

## Run the container on a waggle node

`sudo docker run -it --rm your_docker_hub_user_name/ptzapp:latest -ki -it number_of_iterations -un camera_user_name -pw camera_password -ip camera_ip_address -obj object_name`



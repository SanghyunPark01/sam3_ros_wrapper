image_args="sam3_ros1:latest" # Image

docker_args=""
docker_args+="--gpus=all" # Use GPU
docker_args+=" --device /dev/dri"
docker_args+=" -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute"
docker_args+=" -v /home/psh/workspace:/workspace/" # directory mount Local(/home/psh)-> Docker(/workspace)
docker_args+=" -w /workspace" # Start point at docker

docker_args+=" --privileged -v /dev:/dev --net=host -v /media:/media"
docker_args+=" --volume /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=unix$DISPLAY -e QT_X11_NO_MITSHM=1" # Display
docker_args+=" -e XDG_RUNTIME_DIR=/tmp/runtime-$UID"
docker_args+=" -it $image_args" 

echo "Launching container:"
echo "> docker run $docker_args"
docker run $docker_args
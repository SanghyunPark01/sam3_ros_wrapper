# SAM3 ROS Wrapper
This repository provides a **ROS wrapper for SAM3 (Segment Anything Model 3)**, compatible with **both ROS1 and ROS2**.  
It enables using SAM3-based segmentation within ROS pipelines while keeping the model code and ROS integration loosely coupled.

## Features

- ✅ ROS1 (Noetic) support
- ✅ ROS2 (Humble) support
- ✅ Python-based ROS node
- ✅ Dockerfile for reproducible environment setup
- ✅ **Dynamic prompt update**
- ❌ No pre-built binaries
- ❌ No model weights included

![demo video](https://github.com/user-attachments/assets/09ad7fbb-2ceb-4db3-a005-d4374342f281)

## Requirements
### Common
- NVIDIA GPU (recommended)
- CUDA-compatible NVIDIA driver
- Python 3.8 (ROS1) / 3.10 (ROS2)
- Python 3.12 (SAM3) => need dual python version
- Pre-built SAM3

### ROS
- **ROS1**: Noetic
- **ROS2**: Humble

## Docker Environment(Recommended)
This repository provides Dockerfiles **only for environment setup.**  
  
The Docker images:
- install system dependencies and ROS
- configure CUDA and Python
- pre-built SAM3
- **NOT build ROS packages**
- **NOT download model weights**  

### Build Docker Image  
ROS1:  
```
cd sam3_ros_wrapper/ros/docker
docker build -f Dockfile.ros1 -t sam3_ros1:latest .
```  
ROS2:  
```
cd sam3_ros_wrapper/ros/docker
docker build -f Dockfile.ros2 -t sam3_ros2:latest .
```  

### Run Docker Container
After build,  
```
./ros1_launch.sh
```
or
```
./ros2_launch.sh
```  

## Build
If your environment is ready, build `sam3_ros_wrapper`.  
```
cd (your_ros_worksapce)/src
git clone https://github.com/SanghyunPark01/sam3_ros_wrapper
git clone https://github.com/SanghyunPark01/segmentation_ros_msg
```

- ROS1
    ```
    cd (your_ros1_worksapce)/
    catkin_make
    ```

- ROS2
    ```
    cd (your_ros2_worksapce)/
    colcon build
    ```

## Parameters and API
### Parameters
Parameters are in `ros/config/config.yaml`.  
- `weight_path`: SAM3 weight path
- `default_prompt`: default prompt of SAM3
- `mode`: 
    - When `keep_all`, is used, segmentation is performed on all incoming data.  
    - When `keep_last` is used, segmentation is applied only to the most recent data, which is suitable for real-time systems.
- `input_img_topic`: input image topic name
- `debug/log`: use log
- `debug/vis`: use visualization

### API    
**Input**  
- By using this, the prompt can be changed dynamically during runtime.
- ros topic name: `/sam3_ros_wrapper/api/input/prompt`  

**Output**  
- Result of Segmentation.
- ros topic name: `/sam3_ros_wrapper/api/output/result`

## Example Usage
- **Change Prompt during runtime.**
    - ROS1: script in `ros/ros1/example_change_prompt.py`
        ```
        python3 example_change_prompt.py _prompt:="building"
        ```
    - ROS2: script in `ros/ros2/example_change_prompt.py`
        ```
        python3 example_change_prompt.py --ros-args -p prompt:="building"
        ```
- **Subcription Example(`segmentation_ros_msg`)**
    - ROS1: script in `ros/ros1/example_sub_result.py`
        ```
        python3 example_sub_result.py
        ```
    - ROS2: script in `ros/ros2/example_sub_result.py`
        ```
        ros2 run sam3_ros_wrapper example_sub_result.py
        ```

## License
The license of this repository follows the [SAM3 license](https://github.com/facebookresearch/sam3?tab=License-1-ov-file).
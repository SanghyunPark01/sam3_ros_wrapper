if("$ENV{ROS_VERSION}" STREQUAL "1")
    cmake_minimum_required(VERSION 3.0.2)
    project(sam3_ros_wrapper)

    find_package(catkin REQUIRED COMPONENTS
        rospy
        std_msgs
        sensor_msgs
    )

    catkin_package()  

    catkin_install_python(PROGRAMS
        ros1/sam3_ros1.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
    )
elseif("$ENV{ROS_VERSION}" STREQUAL "2")
    cmake_minimum_required(VERSION 3.5)
    project(sam3_ros_wrapper)

    find_package(ament_cmake REQUIRED)
    find_package(ament_cmake_python REQUIRED)

    find_package(rclpy REQUIRED)
    find_package(std_msgs REQUIRED)
    find_package(sensor_msgs REQUIRED)

    install(
        DIRECTORY ros2/launch ./config
        DESTINATION share/${PROJECT_NAME}
    )

    install(
        DIRECTORY scripts
        DESTINATION lib/${PROJECT_NAME}
    )

    install(PROGRAMS
        ros2/sam3_ros2.py
        ros2/example_sub_result.py
        DESTINATION lib/${PROJECT_NAME}
    )

    ament_package()
endif()
cmake_minimum_required(VERSION 3.15)
project(stereo_depth_estimation)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -diag-suppress=20012")
set(CMAKE_CUDA_ARCHITECTURES 86)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(rclcpp REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(TensorRT REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(spdlog REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(message_filters REQUIRED)

add_executable(stereo_depth src/stereo_depth_node.cpp
                            src/stereo_depth.cpp)
target_include_directories(stereo_depth PRIVATE include
                                            ${OpenCV_INCLUDE_DIRS}
                                            ${CUDAToolkit_INCLUDE_DIR} 
                                            ${TensorRT_INCLUDE_DIRS})
ament_target_dependencies(stereo_depth rclcpp cv_bridge image_transport pcl_ros tf2_ros message_filters)

target_link_libraries(stereo_depth
    ${OpenCV_LIBS}
    spdlog::spdlog_header_only
    Eigen3::Eigen
    ${CUDAToolkit_LIBRARY_DIR}/libcudart.so
    ${TensorRT_LIBRARY_DIRS}/libnvinfer.so)

install(TARGETS stereo_depth
  DESTINATION lib/${PROJECT_NAME})

install(
  DIRECTORY config launch weights
  DESTINATION share/${PROJECT_NAME}
)

ament_package()

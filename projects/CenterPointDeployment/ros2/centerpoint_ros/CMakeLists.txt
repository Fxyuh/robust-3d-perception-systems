cmake_minimum_required(VERSION 3.22)
project(centerpoint_ros LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Detection3D.msg"
  "msg/Detection3DArray.msg"
  "msg/FrameMetadata.msg"
  "msg/TrackedObject3D.msg"
  "msg/TrackedObject3DArray.msg"
  DEPENDENCIES geometry_msgs std_msgs)

add_library(multi_object_tracker STATIC src/multi_object_tracker.cpp)
target_include_directories(multi_object_tracker PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(multi_object_tracker PUBLIC Eigen3::Eigen)
target_compile_options(multi_object_tracker PRIVATE
  -O3 -Wall -Wextra -Wpedantic)

get_filename_component(CPD_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cpp"
  ABSOLUTE)
set(CPD_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
add_subdirectory("${CPD_CORE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/centerpoint_core")

add_executable(centerpoint_node src/centerpoint_node.cpp)
ament_target_dependencies(centerpoint_node
  geometry_msgs rclcpp sensor_msgs std_msgs visualization_msgs)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(centerpoint_node
  centerpoint_detector "${cpp_typesupport_target}")
target_compile_options(centerpoint_node PRIVATE
  -O3 -Wall -Wextra -Wpedantic)

get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
set(CPD_RUNTIME_LIBRARY_DIRS
  "${TENSORRT_LIBRARY_DIR};${CUDAToolkit_ROOT}/lib")
set_target_properties(centerpoint_node PROPERTIES
  BUILD_RPATH "${CPD_RUNTIME_LIBRARY_DIRS}"
  INSTALL_RPATH "${CPD_RUNTIME_LIBRARY_DIRS}")

add_executable(pointcloud_replay_node src/pointcloud_replay_node.cpp)
target_include_directories(pointcloud_replay_node PRIVATE "${CPD_CORE_DIR}")
ament_target_dependencies(pointcloud_replay_node rclcpp sensor_msgs std_msgs)
target_link_libraries(pointcloud_replay_node "${cpp_typesupport_target}")
target_compile_options(pointcloud_replay_node PRIVATE
  -O3 -Wall -Wextra -Wpedantic)

add_executable(centerpoint_tracker_node src/centerpoint_tracker_node.cpp)
ament_target_dependencies(centerpoint_tracker_node
  geometry_msgs rclcpp std_msgs visualization_msgs)
target_link_libraries(centerpoint_tracker_node
  multi_object_tracker "${cpp_typesupport_target}")
target_compile_options(centerpoint_tracker_node PRIVATE
  -O3 -Wall -Wextra -Wpedantic)

install(TARGETS centerpoint_node pointcloud_replay_node centerpoint_tracker_node
  DESTINATION lib/${PROJECT_NAME})
install(TARGETS multi_object_tracker
  ARCHIVE DESTINATION lib)
install(DIRECTORY include/
  DESTINATION include)
install(DIRECTORY launch rviz
  DESTINATION share/${PROJECT_NAME})

include(CTest)
if(BUILD_TESTING)
  add_executable(test_multi_object_tracker test/test_multi_object_tracker.cpp)
  target_link_libraries(test_multi_object_tracker PRIVATE multi_object_tracker)
  target_compile_options(test_multi_object_tracker PRIVATE
    -O2 -Wall -Wextra -Wpedantic)
  add_test(NAME multi_object_tracker COMMAND test_multi_object_tracker)
endif()

ament_export_include_directories(include)
ament_export_dependencies(rosidl_default_runtime)
ament_package()

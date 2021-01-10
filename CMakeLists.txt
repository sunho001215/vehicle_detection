cmake_minimum_required(VERSION 3.10.2)
project(vehicle_detection)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_PREFIX_PATH="/home/sunho/libtorch;/opt/ros/melodic")

set (CMAKE_CXX_STANDARD 14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  roslib
  sensor_msgs
  message_generation
  message_filters
  std_msgs
  geometry_msgs
  tf
  tf2
)

find_package(OpenCV 3 REQUIRED)
find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

catkin_package(
	INCLUDE_DIRS include
	CATKIN_DEPENDS roscpp std_msgs sensor_msgs

	DEPENDS
	opencv_highgui opencv_imgproc
)


include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)
include_directories(include)

set (SOURCES
 src/Darknet.cpp
 src/Visualize.cpp
 src/NMS.cpp
 src/tracking_object.cpp
)

set (HEADERS
 include/Darknet.h
 include/Visualize.h
 include/NMS.h
 include/tracking_object.h
)

add_executable(vehicle_detection_img src/ImgDetection.cpp ${HEADERS} ${SOURCES})
add_dependencies(vehicle_detection_img ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vehicle_detection_img ${OpenCV_LIBS} ${catkin_LIBRARIES}  ${TORCH_LIBRARIES})

add_executable(vehicle_detection_pc src/PCDetection.cpp ${HEADERS} ${SOURCES})
add_dependencies(vehicle_detection_pc ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vehicle_detection_pc ${OpenCV_LIBS} ${catkin_LIBRARIES}  ${TORCH_LIBRARIES})

add_executable(vehicle_tracker src/tracker.cpp ${HEADERS} ${SOURCES})
add_dependencies(vehicle_tracker ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
target_link_libraries(vehicle_tracker ${OpenCV_LIBS} ${catkin_LIBRARIES} Eigen3::Eigen ${TORCH_LIBRARIES})

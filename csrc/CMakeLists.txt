cmake_minimum_required(VERSION 3.10)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 14)

# set the project name
project(vo)

# Add OpenCV
if(APPLE)
	#set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.6.0_1/lib/cmake/opencv4)
	#include_directories(/opt/homebrew/Cellar/opencv/4.6.0_1/include/opencv4)
	set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
	include_directories(/usr/local/include/opencv4)
	link_directories(/usr/local/lib/libopencv_videoio )
	link_directories(/usr/local/lib/libopencv_highgui )
	link_directories(/usr/local/lib/libopencv_features2d )
else()
	set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
	include_directories(/usr/include/opencv4)
	link_directories(/usr/lib/aarch64-linux-gnu/libopencv_videoio )
	link_directories(/usr/lib/aarch64-linux-gnu/libopencv_highgui )
	link_directories(/usr/lib/aarch64-linux-gnu/libopencv_features2d )
endif()

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})


# Eigen
include_directories(./eigen)

# add the executable
add_executable(vo main.cpp)


target_link_libraries(vo ${OPENCV_LIBS} opencv_videoio opencv_highgui opencv_features2d)

cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 11)

# set the project name
project(vo)

# Add OpenCV
set(OpenCV_DIR /opt/homebrew/Cellar/opencv/4.6.0_1/lib/cmake/opencv4)
include_directories(/opt/homebrew/Cellar/opencv/4.6.0_1/include/opencv4)
find_package(OpenCV REQUIRED)

# add the executable
add_executable(vo main.cpp)

target_link_libraries(vo ${OPENCV_LIBS})

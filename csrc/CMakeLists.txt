cmake_minimum_required(VERSION 3.18)  # Bumped for Open3D compatibility

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 17)  # Bumped from 14 — Open3D requires C++17

project(vo)

# ── OpenCV ────────────────────────────────────────────────────────────────────
if(APPLE)
  set(OpenCV_DIR /usr/local/lib/cmake/opencv4)
  include_directories(/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3)
  include_directories(/usr/local/include/opencv4)
else()
  set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
  include_directories(/usr/include/opencv4)
  link_directories(/usr/lib/aarch64-linux-gnu/libopencv_videoio)
  link_directories(/usr/lib/aarch64-linux-gnu/libopencv_highgui)
  link_directories(/usr/lib/aarch64-linux-gnu/libopencv_features2d)
endif()

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

# ── Ceres ─────────────────────────────────────────────────────────────────────
find_package(Ceres REQUIRED)

# ── Eigen (local) ─────────────────────────────────────────────────────────────
include_directories(./eigen)

# ── Open3D ────────────────────────────────────────────────────────────────────
set(Open3D_DIR ${CMAKE_CURRENT_SOURCE_DIR}/open3d/lib/cmake/Open3D)
find_package(Open3D REQUIRED)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(vo main.cpp)

target_link_libraries(vo
  ${OPENCV_LIBS}
  opencv_videoio
  opencv_highgui
  opencv_features2d
  opencv_calib3d
  Ceres::ceres
  Open3D::Open3D
)

cmake_minimum_required(VERSION 3.20)
project(UPnPL)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto")
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS} include)

find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

set(OPENGV_PATH /home/ljj/source_code/opengv)

include_directories(${OPENGV_PATH}/include)
link_directories(${OPENGV_PATH}/build/lib)

add_library(UPnPL src/UPnPL.cc)
target_link_libraries(UPnPL ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(sim_test src/sim_test.cc)
target_link_libraries(sim_test UPnPL opengv)

add_executable(euroc_test src/euroc_test.cc)
target_link_libraries(euroc_test UPnPL yaml-cpp opengv)

add_executable(kitti_test src/kitti_test.cc)
target_link_libraries(kitti_test UPnPL yaml-cpp opengv)


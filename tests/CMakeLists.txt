cmake_minimum_required(VERSION 3.20)
project(UPnPLPointConstraintsTests LANGUAGES CXX)

# Standalone tests need only the solver's dependencies, not the dataset drivers.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto")
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
enable_testing()

set(UPNPL_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
add_library(upnpl_current "${UPNPL_ROOT}/src/UPnPL.cc")
target_include_directories(upnpl_current PUBLIC "${UPNPL_ROOT}/include")
target_link_libraries(upnpl_current PUBLIC Eigen3::Eigen Ceres::ceres)

add_executable(point_constraints_test point_constraints_test.cc)
target_include_directories(point_constraints_test PRIVATE "${UPNPL_ROOT}/include")
target_link_libraries(point_constraints_test PRIVATE Eigen3::Eigen)
add_test(NAME point_constraints COMMAND point_constraints_test)

add_executable(pose_simulation pose_simulation.cc)
target_link_libraries(pose_simulation PRIVATE upnpl_current)
add_test(NAME pose_simulation COMMAND pose_simulation --check)

# Export an unchanged baseline source with git show; never modify the baseline.
set(UPNPL_BASELINE_SOURCE "" CACHE FILEPATH "Original UPnPL.cc for comparison")
if(UPNPL_BASELINE_SOURCE)
    add_library(upnpl_baseline "${UPNPL_BASELINE_SOURCE}")
    target_include_directories(upnpl_baseline PUBLIC "${UPNPL_ROOT}/include")
    target_link_libraries(upnpl_baseline PUBLIC Eigen3::Eigen Ceres::ceres)
    add_executable(pose_simulation_baseline pose_simulation.cc)
    target_link_libraries(pose_simulation_baseline PRIVATE upnpl_baseline)
endif()

option(UPNPL_TEST_RANSAC "Also build comparisons using the existing RANSAC wrapper" OFF)
if(UPNPL_TEST_RANSAC)
    find_package(OpenCV 4 REQUIRED)
    set(OPENGV_PATH "/home/ljj/source_code/opengv" CACHE PATH "OpenGV source/build tree")
    find_library(OPENGV_LIBRARY opengv HINTS "${OPENGV_PATH}/build/lib" REQUIRED)
    foreach(variant IN ITEMS current baseline)
        if(TARGET upnpl_${variant})
            add_executable(pose_ransac_${variant} pose_simulation.cc)
            target_compile_definitions(pose_ransac_${variant} PRIVATE UPNPL_TEST_RANSAC)
            target_include_directories(pose_ransac_${variant} PRIVATE
                "${OPENGV_PATH}/include" ${OpenCV_INCLUDE_DIRS})
            target_link_libraries(pose_ransac_${variant} PRIVATE
                upnpl_${variant} ${OPENGV_LIBRARY} ${OpenCV_LIBS})
        endif()
    endforeach()
endif()

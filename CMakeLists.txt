cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}" ${CMAKE_MODULE_PATH})

if(WIN32)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ../bin)
endif()

if(UNIX AND NOT APPLE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)
endif()

project(segmentation)

find_package(Threads)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

add_executable(segmentation segmentation.cpp)
target_include_directories(segmentation PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS})
target_link_libraries(segmentation PUBLIC ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${TensorRT_LIBRARIES})
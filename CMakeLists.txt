cmake_minimum_required (VERSION 3.1)

project(pyramids)

# Handle OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Handle OpenCV
# set("OpenCV_DIR" "/home/smithzv/tmp/opencv/build-cuda")
# set("OpenCV_DIR" "/home/smithzv/.local/share/OpenCV")
set("OpenCV_DIR" "/usr/share/OpenCV")
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

add_executable (pyramids pyramids.cpp cl-helper.c)
target_include_directories (pyramids PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (pyramids ${OpenCL_LIBRARY} ${OpenCV_LIBS})

add_executable (effects effects.cpp cl-helper.c)
target_include_directories (effects PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (effects ${OpenCL_LIBRARY} ${OpenCV_LIBS})

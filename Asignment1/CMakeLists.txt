cmake_minimum_required(VERSION 3.10)
project(Assignment1)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( Assignment1 Assignment1.cpp )
target_link_libraries( Assignment1 ${OpenCV_LIBS} )
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


cmake_minimum_required(VERSION 3.10)
project(Assignment2)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_executable(Assignment2 Assignment2.cpp)
target_link_libraries (Assignment2 ${OpenCV_LIBS} )

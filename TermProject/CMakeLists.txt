cmake_minimum_required(VERSION 3.10)
project(TermProject)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_executable( TermProject TermProject.cpp )
target_link_libraries( TermProject ${OpenCV_LIBS} )



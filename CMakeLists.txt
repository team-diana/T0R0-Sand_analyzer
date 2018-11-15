cmake_minimum_required(VERSION 2.8)

set (CMAKE_CXX_STANDARD 11)

project(sand)
find_package(OpenCV REQUIRED)
add_executable(sand main.cpp)
target_link_libraries(sand ${OpenCV_LIBS})

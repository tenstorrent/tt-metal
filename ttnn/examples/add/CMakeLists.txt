cmake_minimum_required(VERSION 3.22...3.30)
project(ttnn_example_add)

add_executable(ttnn_example_add)
target_sources(ttnn_example_add PRIVATE add.cpp)

find_package(TT-NN REQUIRED)
target_link_libraries(ttnn_example_add PUBLIC TTNN::TTNN)

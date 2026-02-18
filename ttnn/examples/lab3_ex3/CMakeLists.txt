cmake_minimum_required(VERSION 3.22...4.2)
project(ttnn_example_lab3_ex3)

add_executable(ttnn_example_lab3_ex3)
target_sources(ttnn_example_lab3_ex3 PRIVATE lab3_ex3.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab3_ex3
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

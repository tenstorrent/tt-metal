cmake_minimum_required(VERSION 3.22...3.30)
project(ttnn_example_lab_eltwise_binary)

add_executable(ttnn_example_lab_eltwise_binary)
target_sources(ttnn_example_lab_eltwise_binary PRIVATE lab_eltwise_binary.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab_eltwise_binary
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

cmake_minimum_required(VERSION 3.22...3.30)
project(ttnn_example_lab2_matmul_ex1)

add_executable(ttnn_example_lab2_matmul_ex1)
target_sources(ttnn_example_lab2_matmul_ex1 PRIVATE lab2_matmul_ex1.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab2_matmul_ex1
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

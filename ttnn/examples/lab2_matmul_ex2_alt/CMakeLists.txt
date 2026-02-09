cmake_minimum_required(VERSION 3.22...3.30)
project(ttnn_example_lab2_matmul_ex2_alt)

add_executable(ttnn_example_lab2_matmul_ex2_alt)
target_sources(ttnn_example_lab2_matmul_ex2_alt PRIVATE lab2_matmul_ex2_alt.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab2_matmul_ex2_alt
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

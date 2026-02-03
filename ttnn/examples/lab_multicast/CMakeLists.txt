cmake_minimum_required(VERSION 3.22...4.2)
project(ttnn_example_lab_multicast)

add_executable(ttnn_example_lab_multicast)
target_sources(ttnn_example_lab_multicast PRIVATE lab_multicast.cpp)

find_package(TT-Metalium REQUIRED)
find_package(TT-NN REQUIRED)

target_link_libraries(
    ttnn_example_lab_multicast
    PUBLIC
        TT::Metalium
        TTNN::TTNN
)

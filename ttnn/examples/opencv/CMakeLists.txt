cmake_minimum_required(VERSION 3.22...3.30)
project(ttnn_example_opencv)

add_executable(ttnn_example_opencv)
target_sources(ttnn_example_opencv PRIVATE opencv.cpp)

find_package(TT-NN REQUIRED)
find_package(OpenCV REQUIRED)

target_link_libraries(
    ttnn_example_opencv
    PUBLIC
        TTNN::TTNN
        ${OpenCV_LIBS}
)
target_include_directories(ttnn_example_opencv PRIVATE ${OpenCV_INCLUDE_DIRS})
#target_link_libraries(ttnn_example_opencv PUBLIC TTNN::TTNN)

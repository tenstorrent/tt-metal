cmake_minimum_required(VERSION 3.22...3.30)

include(${PROJECT_SOURCE_DIR}/cmake/helper_functions.cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tt_telemetry")

project(tt_telemetry)

link_libraries(TT::STL)

add_executable(tt_telemetry_server)
target_sources(tt_telemetry_server PRIVATE server/main.cpp)

if(NOT TARGET TT::Metalium)
    find_package(TT-Metalium REQUIRED)
endif()
target_link_libraries(tt_telemetry_server PUBLIC TT::Metalium)
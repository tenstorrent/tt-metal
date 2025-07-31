cmake_minimum_required(VERSION 3.22...3.30)

add_library(telemetry_libs INTERFACE)
target_include_directories(telemetry_libs INTERFACE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_link_libraries(
    telemetry_libs
    INTERFACE
        pthread
        nlohmann_json::nlohmann_json
        magic_enum::magic_enum
        fmt::fmt-header-only
        small_vector
        Boost::algorithm
        Taskflow::Taskflow
        TT::Metalium
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tt_telemetry")

add_executable(tt_telemetry_server)
target_sources(tt_telemetry_server PRIVATE server/main.cpp)
target_include_directories(
    tt_telemetry_server
    PRIVATE
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(
    tt_telemetry_server
    PRIVATE
        tt_metal
        telemetry_libs
)
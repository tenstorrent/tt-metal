cmake_minimum_required(VERSION 3.22...3.30)

# Find required packages
find_package(Threads REQUIRED)

# Fetch cpp-httplib (much simpler, no complex dependencies)
include(FetchContent)
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.14.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(httplib)

add_library(telemetry_libs INTERFACE)
target_include_directories(telemetry_libs INTERFACE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_link_libraries(
    telemetry_libs
    INTERFACE
        Threads::Threads
        nlohmann_json::nlohmann_json
        fmt::fmt-header-only
        small_vector
        Boost::algorithm
        Taskflow::Taskflow
        TT::Metalium
        # httplib will be linked directly to the executable
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tt_telemetry")

add_executable(tt_telemetry_server)
target_sources(
    tt_telemetry_server
    PRIVATE
        server/main.cpp
        server/telemetry_provider.cpp
        server/web_server.cpp
        telemetry/ethernet/chip_identifier.cpp
        telemetry/ethernet/ethernet_endpoint.cpp
        telemetry/ethernet/ethernet_link.cpp
        telemetry/ethernet/ethernet_helpers.cpp
        telemetry/ethernet/print_helpers.cpp
        telemetry/ethernet/ethernet_metrics.cpp
)
target_include_directories(
    tt_telemetry_server
    PRIVATE
        include
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
)
target_link_libraries(
    tt_telemetry_server
    PRIVATE
        tt_metal
        telemetry_libs
        httplib::httplib  # Link httplib only to the server executable
)
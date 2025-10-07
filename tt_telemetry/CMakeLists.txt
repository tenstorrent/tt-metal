cmake_minimum_required(VERSION 3.22...3.30)

# Find required packages
find_package(Threads REQUIRED)

# Fetch cpp-httplib using CPM (consistent with third_party dependencies)
CPMAddPackage(
    NAME httplib
    GITHUB_REPOSITORY yhirose/cpp-httplib
    GIT_TAG v0.14.1
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)

add_library(telemetry_libs INTERFACE)
target_include_directories(telemetry_libs INTERFACE "$<TARGET_PROPERTY:TT::Metalium,INCLUDE_DIRECTORIES>")
target_link_libraries(
    telemetry_libs
    INTERFACE
        Threads::Threads
        nlohmann_json::nlohmann_json
        fmt::fmt-header-only
        Boost::algorithm
        Taskflow::Taskflow
    # We use some TT::Metalium headers but do not need to link the library in
    # httplib will be linked directly to the executable
)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/tt_telemetry")

add_executable(tt_telemetry_server)
target_sources(
    tt_telemetry_server
    PRIVATE
        server/main.cpp
        telemetry/telemetry_provider.cpp
        telemetry/metric.cpp
        server/web_server.cpp
        telemetry/ethernet/chip_identifier.cpp
        telemetry/ethernet/ethernet_endpoint.cpp
        telemetry/ethernet/ethernet_helpers.cpp
        telemetry/ethernet/ethernet_metrics.cpp
        telemetry/arc/arc_telemetry_reader.cpp
        telemetry/arc/arc_metrics.cpp
        telemetry/hal/hal.cpp
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
        httplib::httplib # Link httplib only to the server executable
        cxxopts::cxxopts
)

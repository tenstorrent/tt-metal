cmake_minimum_required(VERSION 3.22...3.30)

# Add local cmake modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Find required packages
find_package(Threads REQUIRED)

# Use gRPC and its bundled Protobuf from third_party (fetched via CPM)
# gRPC bundles its own protobuf to keep it isolated from tt-metal's protobuf
# Set up the variables expected by BuildTelemetryProto.cmake
# Use grpc++_unsecure since telemetry server uses UNIX sockets with InsecureServerCredentials (no SSL needed)
set(_PROTOBUF_LIBPROTOBUF libprotobuf)  # gRPC's bundled protobuf (no protobuf:: namespace)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)  # gRPC's bundled protoc
set(_GRPC_GRPCPP grpc++_unsecure)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)
set(_REFLECTION grpc++_reflection)
set(GRPC_FOUND_VIA_CMAKE TRUE)

# Fetch cpp-httplib using CPM (consistent with third_party dependencies)
CPMAddPackage(
    NAME httplib
    GITHUB_REPOSITORY yhirose/cpp-httplib
    GIT_TAG v0.14.1
    OPTIONS
        "CMAKE_MESSAGE_LOG_LEVEL NOTICE"
)

# Fetch websocketpp using CPM
CPMAddPackage(
    NAME websocketpp
    GITHUB_REPOSITORY zaphoyd/websocketpp
    GIT_TAG 0.8.2
    DOWNLOAD_ONLY YES
    PATCHES
        websocketpp_cpp_cleanup.patch
)

# Create interface library for websocketpp
if(websocketpp_ADDED)
    add_library(websocketpp INTERFACE)
    target_include_directories(websocketpp INTERFACE ${websocketpp_SOURCE_DIR})

    # websocketpp requires these system libraries
    find_package(Threads REQUIRED)
    target_link_libraries(websocketpp INTERFACE Threads::Threads)

    # websocketpp can use Boost.Asio or standalone asio
    # We'll use standalone asio to avoid Boost dependency
    target_compile_definitions(
        websocketpp
        INTERFACE
            ASIO_STANDALONE
            _WEBSOCKETPP_CPP11_STL_
            _WEBSOCKETPP_CPP11_FUNCTIONAL_
    )
endif()

# Fetch standalone asio for websocketpp
CPMAddPackage(NAME asio GITHUB_REPOSITORY chriskohlhoff/asio GIT_TAG asio-1-24-0 DOWNLOAD_ONLY YES)

if(asio_ADDED)
    add_library(asio INTERFACE)
    target_include_directories(asio INTERFACE ${asio_SOURCE_DIR}/asio/include)
    target_compile_definitions(asio INTERFACE ASIO_STANDALONE)
endif()

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

# Build the telemetry gRPC proto library
include(BuildTelemetryProto)

add_executable(tt_telemetry_server)
target_sources(
    tt_telemetry_server
    PRIVATE
        server/main.cpp
        telemetry/telemetry_collector.cpp
        telemetry/telemetry_subscriber.cpp
        telemetry/watchdog.cpp
        telemetry/metric.cpp
        server/web_server.cpp
        server/prom_formatter.cpp
        server/collection_endpoint.cpp
        server/collection_clients.cpp
        server/grpc_telemetry_server.cpp
        topology/topology.cpp
        telemetry/ethernet/ethernet_helpers.cpp
        telemetry/ethernet/ethernet_metrics.cpp
        telemetry/arc/arc_metrics.cpp
        telemetry/system/system_metrics.cpp
        telemetry/hal/hal.cpp
)
target_include_directories(
    tt_telemetry_server
    PRIVATE
        include
        ${UMD_HOME}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${CMAKE_CURRENT_SOURCE_DIR}/common
        ${CMAKE_BINARY_DIR}/tools/scaleout # For generated protobuf headers
)
target_link_libraries(
    tt_telemetry_server
    PRIVATE
        tt_metal
        telemetry_libs
        httplib::httplib # Link httplib only to the server executable
        websocketpp # Link websocketpp for WebSocket functionality
        asio # Standalone asio for websocketpp
        cxxopts::cxxopts
        ${_PROTOBUF_LIBPROTOBUF} # Add protobuf library for compatibility
        telemetry_grpc_proto # Link the generated gRPC proto library
        ${_GRPC_GRPCPP}
)

# Copy frontend files to build directory so they're accessible relative to the executable
# Validate that frontend directory exists
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static")
    message(FATAL_ERROR "Frontend static directory not found at ${CMAKE_CURRENT_SOURCE_DIR}/frontend/static")
endif()

# Track frontend source files for dependency tracking
file(GLOB_RECURSE FRONTEND_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/frontend/static/*")

# Create a custom target to manage frontend asset copying with proper dependencies
add_custom_target(
    copy_frontend_assets
    ALL
    COMMAND
        ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/frontend/static
        ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/frontend/static
    DEPENDS
        ${FRONTEND_SOURCES}
    COMMENT "Copying frontend files to build directory"
)

# Make the server target depend on frontend assets so they're always up-to-date
add_dependencies(tt_telemetry_server copy_frontend_assets)

# Add tests subdirectory
if(TT_METAL_BUILD_TESTS)
    add_subdirectory(tests)
endif()

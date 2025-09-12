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

# Fetch uWebSockets using CPM
CPMAddPackage(NAME uWebSockets GITHUB_REPOSITORY uNetworking/uWebSockets GIT_TAG v20.67.0 DOWNLOAD_ONLY YES)

# Fetch uSockets (dependency of uWebSockets) using CPM
CPMAddPackage(NAME uSockets GITHUB_REPOSITORY uNetworking/uSockets GIT_TAG v0.8.8 DOWNLOAD_ONLY YES)

# Create interface libraries for uWebSockets and uSockets
if(uWebSockets_ADDED AND uSockets_ADDED)
    # Check if we have the required source files
    set(USOCKETS_SOURCES
        ${uSockets_SOURCE_DIR}/src/bsd.c
        ${uSockets_SOURCE_DIR}/src/context.c
        ${uSockets_SOURCE_DIR}/src/loop.c
        ${uSockets_SOURCE_DIR}/src/socket.c
    )

    # Add platform-specific eventing
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        list(APPEND USOCKETS_SOURCES ${uSockets_SOURCE_DIR}/src/eventing/epoll_kqueue.c)
        set(USOCKETS_DEFS LIBUS_USE_EPOLL)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        list(APPEND USOCKETS_SOURCES ${uSockets_SOURCE_DIR}/src/eventing/epoll_kqueue.c)
        set(USOCKETS_DEFS LIBUS_USE_KQUEUE)
    else()
        # Fallback to libuv for other platforms
        list(APPEND USOCKETS_SOURCES ${uSockets_SOURCE_DIR}/src/eventing/libuv.c)
        set(USOCKETS_DEFS LIBUS_USE_LIBUV)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(LIBUV REQUIRED libuv)
    endif()

    # uSockets library
    add_library(uSockets STATIC ${USOCKETS_SOURCES})
    target_include_directories(uSockets PUBLIC ${uSockets_SOURCE_DIR}/src)
    target_compile_definitions(
        uSockets
        PRIVATE
            ${USOCKETS_DEFS}
            LIBUS_NO_SSL
    )

    # Suppress warnings for uSockets third-party code (C library)
    target_compile_options(
        uSockets
        PRIVATE
            $<$<C_COMPILER_ID:Clang>:-Wno-conditional-uninitialized>
            $<$<C_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
            $<$<C_COMPILER_ID:Clang>:-Wno-error>
            $<$<C_COMPILER_ID:GNU>:-Wno-error>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-conditional-uninitialized>
            $<$<CXX_COMPILER_ID:GNU>:-Wno-maybe-uninitialized>
            $<$<CXX_COMPILER_ID:Clang>:-Wno-error>
            $<$<CXX_COMPILER_ID:GNU>:-Wno-error>
    )

    # Add libuv if needed
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR USOCKETS_DEFS MATCHES "LIBUS_USE_LIBUV")
        target_link_libraries(uSockets PRIVATE ${LIBUV_LIBRARIES})
        target_include_directories(uSockets PRIVATE ${LIBUV_INCLUDE_DIRS})
    endif()

    # Only add OpenSSL support if available
    if(WEBSOCKET_SSL_SUPPORT)
        target_sources(uSockets PRIVATE ${uSockets_SOURCE_DIR}/src/crypto/openssl.c)
        target_compile_definitions(uSockets PRIVATE LIBUS_USE_OPENSSL)
        # Remove LIBUS_NO_SSL
        get_target_property(CURRENT_DEFS uSockets COMPILE_DEFINITIONS)
        list(REMOVE_ITEM CURRENT_DEFS LIBUS_NO_SSL)
        set_target_properties(
            uSockets
            PROPERTIES
                COMPILE_DEFINITIONS
                    "${CURRENT_DEFS}"
        )
    endif()

    # uWebSockets interface library
    add_library(uWebSockets INTERFACE)
    target_include_directories(uWebSockets INTERFACE ${uWebSockets_SOURCE_DIR}/src)
    target_link_libraries(uWebSockets INTERFACE uSockets)
endif()

# uWebSockets requires additional system libraries
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZLIB REQUIRED zlib)

# OpenSSL is optional - only needed for SSL/TLS support in uWebSockets
find_package(OpenSSL QUIET)
if(OpenSSL_FOUND)
    message(STATUS "OpenSSL found - SSL/TLS support enabled for WebSocket server")
    set(WEBSOCKET_SSL_SUPPORT ON)
else()
    message(STATUS "OpenSSL not found - WebSocket server will run without SSL/TLS support")
    set(WEBSOCKET_SSL_SUPPORT OFF)
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

add_executable(tt_telemetry_server)
target_sources(
    tt_telemetry_server
    PRIVATE
        server/main.cpp
        telemetry/telemetry_provider.cpp
        telemetry/metric.cpp
        server/web_server.cpp
        server/websocket_server.cpp
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
        ${ZLIB_INCLUDE_DIRS} # Required by uWebSockets
)
target_link_libraries(
    tt_telemetry_server
    PRIVATE
        tt_metal
        telemetry_libs
        httplib::httplib # Link httplib only to the server executable
        uWebSockets # Link uWebSockets for WebSocket functionality
        ${ZLIB_LIBRARIES} # Required by uWebSockets
        cxxopts::cxxopts
)

# Link uSockets dependencies
if(uWebSockets_ADDED AND uSockets_ADDED)
    target_link_libraries(uSockets PRIVATE ${ZLIB_LIBRARIES})
    if(WEBSOCKET_SSL_SUPPORT)
        target_link_libraries(
            uSockets
            PRIVATE
                OpenSSL::SSL
                OpenSSL::Crypto
        )
        target_compile_definitions(uSockets PRIVATE LIBUS_USE_OPENSSL)
    endif()
endif()

# Conditionally link OpenSSL if available
if(WEBSOCKET_SSL_SUPPORT)
    target_compile_definitions(tt_telemetry_server PRIVATE WEBSOCKET_SSL_SUPPORT=1)
endif()

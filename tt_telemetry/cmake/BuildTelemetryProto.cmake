# BuildTelemetryProto.cmake
#
# Builds the telemetry gRPC proto library from telemetry_service.proto
#
# This module expects the following variables to be set (by FindSystemgRPC.cmake):
#   _PROTOBUF_LIBPROTOBUF - Protobuf library target
#   _PROTOBUF_PROTOC - protoc executable
#   _GRPC_GRPCPP - gRPC C++ library target
#   _GRPC_CPP_PLUGIN_EXECUTABLE - grpc_cpp_plugin executable
#   GRPC_FOUND_VIA_CMAKE - Boolean indicating detection method
#
# Creates the following target:
#   telemetry_grpc_proto - Library containing generated protobuf/gRPC code

# Proto file location
set(TELEMETRY_PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/server/telemetry_service.proto)

# Create library target
add_library(telemetry_grpc_proto ${TELEMETRY_PROTO_FILES})

# Link against gRPC and Protobuf
target_link_libraries(
    telemetry_grpc_proto
    PUBLIC
        ${_PROTOBUF_LIBPROTOBUF}
        ${_GRPC_GRPCPP}
)

# Make generated headers available
target_include_directories(telemetry_grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

# Generate C++ files from proto
if(GRPC_FOUND_VIA_CMAKE)
    # Modern CMake approach (Ubuntu 24.04+)
    message(STATUS "Generating telemetry proto files using modern CMake...")

    protobuf_generate(
        TARGET telemetry_grpc_proto
        LANGUAGE cpp
    )

    protobuf_generate(
        TARGET telemetry_grpc_proto
        LANGUAGE grpc
        GENERATE_EXTENSIONS .grpc.pb.h .grpc.pb.cc
        PLUGIN "protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}"
    )
else()
    # pkg-config approach (Ubuntu 22.04)
    message(STATUS "Generating telemetry proto files using pkg-config method...")

    # Verify we have the required tools
    if(NOT _PROTOBUF_PROTOC)
        message(FATAL_ERROR "_PROTOBUF_PROTOC is not set. FindSystemgRPC may have failed.")
    endif()
    if(NOT _GRPC_CPP_PLUGIN_EXECUTABLE)
        message(FATAL_ERROR "_GRPC_CPP_PLUGIN_EXECUTABLE is not set. FindSystemgRPC may have failed.")
    endif()

    message(STATUS "Using protoc: ${_PROTOBUF_PROTOC}")
    message(STATUS "Using grpc_cpp_plugin: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")

    # Get the proto file path
    get_target_property(proto_sources telemetry_grpc_proto SOURCES)

    # Manually generate protobuf and gRPC C++ files
    set(PROTO_SRCS)
    set(PROTO_HDRS)
    set(GRPC_SRCS)
    set(GRPC_HDRS)

    foreach(proto_file ${proto_sources})
        get_filename_component(proto_file_abs ${proto_file} ABSOLUTE)
        get_filename_component(proto_file_we ${proto_file} NAME_WE)
        get_filename_component(proto_path ${proto_file_abs} PATH)

        set(proto_src "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.pb.cc")
        set(proto_hdr "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.pb.h")
        set(grpc_src "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.grpc.pb.cc")
        set(grpc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.grpc.pb.h")

        # Generate protobuf C++ files
        add_custom_command(
            OUTPUT
                ${proto_src}
                ${proto_hdr}
            COMMAND
                ${_PROTOBUF_PROTOC}
            ARGS
                --cpp_out=${CMAKE_CURRENT_BINARY_DIR}
                -I${proto_path} ${proto_file_abs}
            DEPENDS
                ${proto_file_abs}
            COMMENT "Generating protobuf C++ code for ${proto_file}"
            VERBATIM
        )

        # Generate gRPC C++ files
        add_custom_command(
            OUTPUT
                ${grpc_src}
                ${grpc_hdr}
            COMMAND
                ${_PROTOBUF_PROTOC}
            ARGS
                --grpc_out=${CMAKE_CURRENT_BINARY_DIR}
                --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
                -I${proto_path} ${proto_file_abs}
            DEPENDS
                ${proto_file_abs}
            COMMENT "Generating gRPC C++ code for ${proto_file}"
            VERBATIM
        )

        list(APPEND PROTO_SRCS ${proto_src})
        list(APPEND PROTO_HDRS ${proto_hdr})
        list(APPEND GRPC_SRCS ${grpc_src})
        list(APPEND GRPC_HDRS ${grpc_hdr})
    endforeach()

    # Add generated sources to the library
    target_sources(
        telemetry_grpc_proto
        PRIVATE
            ${PROTO_SRCS}
            ${PROTO_HDRS}
            ${GRPC_SRCS}
            ${GRPC_HDRS}
    )
endif()

message(STATUS "Telemetry proto library configured: telemetry_grpc_proto")

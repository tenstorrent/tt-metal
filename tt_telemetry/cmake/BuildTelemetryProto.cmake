# BuildTelemetryProto.cmake
#
# Builds the telemetry gRPC proto library from telemetry_service.proto
#
# This module expects the following variables to be set (by FetchgRPC.cmake):
#   _PROTOBUF_LIBPROTOBUF - Protobuf library target
#   _PROTOBUF_PROTOC - protoc executable
#   _GRPC_GRPCPP - gRPC C++ library target
#   _GRPC_CPP_PLUGIN_EXECUTABLE - grpc_cpp_plugin executable
#
# Creates the following target:
#   telemetry_grpc_proto - Library containing generated protobuf/gRPC code

# Proto file location
set(TELEMETRY_PROTO_FILES ${CMAKE_CURRENT_SOURCE_DIR}/include/server/telemetry_service.proto)

# Verify we have the required tools
if(NOT _PROTOBUF_PROTOC)
    message(FATAL_ERROR "_PROTOBUF_PROTOC is not set. FetchgRPC may have failed.")
endif()
if(NOT _GRPC_CPP_PLUGIN_EXECUTABLE)
    message(FATAL_ERROR "_GRPC_CPP_PLUGIN_EXECUTABLE is not set. FetchgRPC may have failed.")
endif()

message(STATUS "Using protoc: ${_PROTOBUF_PROTOC}")
message(STATUS "Using grpc_cpp_plugin: ${_GRPC_CPP_PLUGIN_EXECUTABLE}")

# Generate protobuf and gRPC C++ files using custom commands
# This avoids using FindProtobuf which would contaminate global state
set(PROTO_SRCS)
set(PROTO_HDRS)
set(GRPC_SRCS)
set(GRPC_HDRS)

foreach(proto_file ${TELEMETRY_PROTO_FILES})
    get_filename_component(proto_file_abs ${proto_file} ABSOLUTE)
    get_filename_component(proto_file_we ${proto_file} NAME_WE)
    get_filename_component(proto_path ${proto_file_abs} PATH)

    # Protobuf outputs
    set(pb_src "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.pb.cc")
    set(pb_hdr "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.pb.h")

    # gRPC outputs
    set(grpc_src "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.grpc.pb.cc")
    set(grpc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${proto_file_we}.grpc.pb.h")

    # Generate protobuf C++ files
    add_custom_command(
        OUTPUT
            ${pb_src}
            ${pb_hdr}
        COMMAND
            ${_PROTOBUF_PROTOC}
        ARGS
            --cpp_out=${CMAKE_CURRENT_BINARY_DIR} -I${proto_path} ${proto_file_abs}
        DEPENDS
            ${proto_file_abs}
            grpc_external
        COMMENT "Generating Protobuf C++ code for ${proto_file}"
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
            --grpc_out=${CMAKE_CURRENT_BINARY_DIR} --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
            -I${proto_path} ${proto_file_abs}
        DEPENDS
            ${proto_file_abs}
            grpc_external
        COMMENT "Generating gRPC C++ code for ${proto_file}"
        VERBATIM
    )

    list(APPEND PROTO_SRCS ${pb_src})
    list(APPEND PROTO_HDRS ${pb_hdr})
    list(APPEND GRPC_SRCS ${grpc_src})
    list(APPEND GRPC_HDRS ${grpc_hdr})
endforeach()

# Create library target with generated sources
add_library(
    telemetry_grpc_proto
    ${PROTO_SRCS}
    ${PROTO_HDRS}
    ${GRPC_SRCS}
    ${GRPC_HDRS}
)

# Ensure gRPC external project is built before this library
if(TARGET grpc_external)
    add_dependencies(telemetry_grpc_proto grpc_external)
endif()

# Link against gRPC and Protobuf
target_link_libraries(
    telemetry_grpc_proto
    PUBLIC
        ${_PROTOBUF_LIBPROTOBUF}
        ${_GRPC_GRPCPP}
)

# Make generated headers available
target_include_directories(telemetry_grpc_proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

message(STATUS "Telemetry proto library configured: telemetry_grpc_proto")

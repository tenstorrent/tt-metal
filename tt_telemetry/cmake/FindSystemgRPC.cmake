# FindSystemgRPC.cmake
#
# Find system-installed gRPC and Protobuf packages.
# This is much faster than building from source in CI/CD.
#
# Supports both:
# - Modern systems (Ubuntu 24.04+) with CMake config files
# - Older systems (Ubuntu 22.04) via pkg-config
#
# Sets the following variables:
#   _PROTOBUF_LIBPROTOBUF - Protobuf library target
#   _PROTOBUF_PROTOC - protoc executable
#   _GRPC_GRPCPP - gRPC C++ library target
#   _GRPC_CPP_PLUGIN_EXECUTABLE - grpc_cpp_plugin executable
#   _REFLECTION - gRPC reflection library (if available)

# Try modern CMake config first (Ubuntu 24.04+)
find_package(gRPC CONFIG QUIET)

if(gRPC_FOUND)
    # Modern system with CMake config files
    find_package(Protobuf REQUIRED)
    message(STATUS "Using system gRPC (CMake): ${gRPC_VERSION}")
    message(STATUS "Using system Protobuf (CMake): ${Protobuf_VERSION}")

    set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
    set(_REFLECTION gRPC::grpc++_reflection)
    set(_PROTOBUF_PROTOC $<TARGET_FILE:protobuf::protoc>)
    set(_GRPC_GRPCPP gRPC::grpc++)
    set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)

    # Export that we found it via CMake
    set(GRPC_FOUND_VIA_CMAKE TRUE)
else()
    # Fall back to pkg-config for older systems (Ubuntu 22.04)
    message(STATUS "CMake config not found, using pkg-config...")
    find_package(PkgConfig REQUIRED)

    pkg_check_modules(GRPC REQUIRED grpc++)
    pkg_check_modules(PROTOBUF REQUIRED protobuf)

    message(STATUS "Using system gRPC (pkg-config): ${GRPC_VERSION}")
    message(STATUS "Using system Protobuf (pkg-config): ${PROTOBUF_VERSION}")

    # Find protoc and grpc_cpp_plugin executables
    find_program(_PROTOBUF_PROTOC_EXEC protoc REQUIRED)
    find_program(_GRPC_CPP_PLUGIN_EXEC grpc_cpp_plugin REQUIRED)

    # Include FindProtobuf to get protobuf_generate command
    include(FindProtobuf)

    # Create imported targets for compatibility
    if(NOT TARGET protobuf::libprotobuf)
        add_library(protobuf::libprotobuf INTERFACE IMPORTED)
        target_link_libraries(protobuf::libprotobuf INTERFACE ${PROTOBUF_LINK_LIBRARIES})
        target_include_directories(protobuf::libprotobuf INTERFACE ${PROTOBUF_INCLUDE_DIRS})
        target_compile_options(protobuf::libprotobuf INTERFACE ${PROTOBUF_CFLAGS_OTHER})
    endif()

    if(NOT TARGET grpc++)
        add_library(grpc++ INTERFACE IMPORTED)
        target_link_libraries(grpc++ INTERFACE ${GRPC_LINK_LIBRARIES})
        target_include_directories(grpc++ INTERFACE ${GRPC_INCLUDE_DIRS})
        target_compile_options(grpc++ INTERFACE ${GRPC_CFLAGS_OTHER})
    endif()

    set(_PROTOBUF_LIBPROTOBUF protobuf::libprotobuf)
    set(_REFLECTION grpc++)
    set(_PROTOBUF_PROTOC ${_PROTOBUF_PROTOC_EXEC})
    set(_GRPC_GRPCPP grpc++)
    set(_GRPC_CPP_PLUGIN_EXECUTABLE ${_GRPC_CPP_PLUGIN_EXEC})

    # Export that we found it via pkg-config
    set(GRPC_FOUND_VIA_CMAKE FALSE)
endif()

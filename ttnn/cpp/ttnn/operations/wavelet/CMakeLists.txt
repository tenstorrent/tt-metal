include(sources.cmake)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(WAVELET_SCHEME_CATALOG ${CMAKE_CURRENT_SOURCE_DIR}/schemes/wavelet_schemes.json)
set(WAVELET_SCHEME_GENERATOR ${CMAKE_CURRENT_SOURCE_DIR}/schemes/generate_wavelet_scheme_headers.py)
set(WAVELET_GENERATED_INCLUDE_ROOT ${PROJECT_BINARY_DIR}/generated/wavelet)
set(WAVELET_GENERATED_HOST_INCLUDE_DIR ${WAVELET_GENERATED_INCLUDE_ROOT}/ttnn/cpp)
set(WAVELET_GENERATED_SCHEME_DIR
    ${WAVELET_GENERATED_HOST_INCLUDE_DIR}/ttnn/operations/wavelet/generated/wavelet_schemes
)

file(READ ${WAVELET_SCHEME_CATALOG} WAVELET_SCHEME_CATALOG_JSON)
string(JSON WAVELET_SCHEME_COUNT LENGTH "${WAVELET_SCHEME_CATALOG_JSON}")
if(NOT WAVELET_SCHEME_COUNT EQUAL 106)
    message(FATAL_ERROR "Expected 106 Wavelet schemes, found ${WAVELET_SCHEME_COUNT}")
endif()

set(WAVELET_GENERATED_SCHEME_HEADERS)
math(EXPR WAVELET_SCHEME_LAST_INDEX "${WAVELET_SCHEME_COUNT} - 1")
foreach(SCHEME_INDEX RANGE 0 ${WAVELET_SCHEME_LAST_INDEX})
    string(JSON SCHEME_NAME MEMBER "${WAVELET_SCHEME_CATALOG_JSON}" ${SCHEME_INDEX})
    string(REPLACE "." "_" SCHEME_IDENTIFIER ${SCHEME_NAME})
    list(APPEND WAVELET_GENERATED_SCHEME_HEADERS ${WAVELET_GENERATED_SCHEME_DIR}/${SCHEME_IDENTIFIER}.hpp)
endforeach()
list(
    APPEND
    WAVELET_GENERATED_SCHEME_HEADERS
    ${WAVELET_GENERATED_SCHEME_DIR}/scheme_catalog.hpp
    ${WAVELET_GENERATED_SCHEME_DIR}/scheme_dispatch.hpp
)
unset(WAVELET_SCHEME_CATALOG_JSON)

add_custom_command(
    OUTPUT
        ${WAVELET_GENERATED_SCHEME_HEADERS}
    COMMAND
        ${Python3_EXECUTABLE} ${WAVELET_SCHEME_GENERATOR} ${WAVELET_GENERATED_SCHEME_DIR}
    DEPENDS
        ${WAVELET_SCHEME_GENERATOR}
        ${WAVELET_SCHEME_CATALOG}
    COMMENT "Generating TTNN Wavelet scheme headers"
    VERBATIM
)

set_source_files_properties(
    ${WAVELET_GENERATED_SCHEME_HEADERS}
    PROPERTIES
        GENERATED
            TRUE
)

add_library(ttnn_op_wavelet ${LIB_TYPE})
add_library(TTNN::Ops::Wavelet ALIAS ttnn_op_wavelet)

tt_reuse_precompile_headers(ttnn_op_wavelet TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_wavelet)

set_target_properties(
    ttnn_op_wavelet
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    common/*.hpp
    device/kernels/*.cpp
    device/kernels/*.h
    device/kernels/*.hpp
    device/protocol/*.hpp
    planner/static_scheme.hpp
    planner/step.hpp
)

target_sources(
    ttnn_op_wavelet
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_WAVELET_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
        FILE_SET generated_schemes
        TYPE HEADERS
        BASE_DIRS ${WAVELET_GENERATED_HOST_INCLUDE_DIR}
        FILES ${WAVELET_GENERATED_SCHEME_HEADERS}
    PRIVATE
        ${TTNN_OP_WAVELET_SRCS}
)

# The implementation sources instantiate all 106 static schemes. Compile them
# outside the unity source to avoid anonymous-namespace collisions and an
# excessively large translation unit.
set_source_files_properties(
    device/wavelet_1d_operation_impl.cpp
    device/wavelet_2d_operation_impl.cpp
    PROPERTIES
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

target_include_directories(ttnn_op_wavelet BEFORE PRIVATE ${WAVELET_GENERATED_HOST_INCLUDE_DIR})
target_include_directories(ttnn_op_wavelet PRIVATE ${FixmeOpIncDirs})
target_compile_definitions(
    ttnn_op_wavelet
    PRIVATE
        TTNN_WAVELET_GENERATED_INCLUDE_ROOT="${WAVELET_GENERATED_INCLUDE_ROOT}"
)
target_link_libraries(ttnn_op_wavelet PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_wavelet
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/wavelet
        COMPONENT ttnn-runtime
    FILE_SET
    generated_schemes
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_wavelet LIBRARY COMPONENT tar)

if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_WAVELET_NANOBIND_SRCS})
endif()

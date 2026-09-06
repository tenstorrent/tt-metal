include(sources.cmake)

add_library(ttnn_op_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Matmul ALIAS ttnn_op_matmul)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(MATMUL_REGISTRY_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/device/config/registry)
set(MATMUL_REGISTRY_GENERATED_DIR ${MATMUL_REGISTRY_SOURCE_DIR}/checked_in)
set(MATMUL_REGISTRY_GENERATED_SOURCE ${MATMUL_REGISTRY_GENERATED_DIR}/matmul_registry_data.cpp)
set(MATMUL_REGISTRY_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/matmul_registry)
set(MATMUL_REGISTRY_BUILD_HEADER ${MATMUL_REGISTRY_BUILD_DIR}/matmul_registry_build_attestation.hpp)
set(MATMUL_REGISTRY_BUILD_EMITTER ${MATMUL_REGISTRY_SOURCE_DIR}/codegen/emit_build_attestation.py)
set(MATMUL_REGISTRY_SEMANTIC_MANIFEST ${MATMUL_REGISTRY_SOURCE_DIR}/semantic_dependencies.txt)

file(STRINGS ${MATMUL_REGISTRY_SEMANTIC_MANIFEST} MATMUL_REGISTRY_SEMANTIC_PATHS)
list(FILTER MATMUL_REGISTRY_SEMANTIC_PATHS EXCLUDE REGEX "^[ \t]*(#|$)")
set(MATMUL_REGISTRY_SEMANTIC_DEPENDENCIES)
foreach(path IN LISTS MATMUL_REGISTRY_SEMANTIC_PATHS)
    list(APPEND MATMUL_REGISTRY_SEMANTIC_DEPENDENCIES ${PROJECT_SOURCE_DIR}/${path})
endforeach()

set(MATMUL_REGISTRY_BUILD_FACT_ARGUMENTS)
foreach(
    fact
    IN
    ITEMS
        cmake_build_type=${CMAKE_BUILD_TYPE}
        cmake_cxx_compiler_id=${CMAKE_CXX_COMPILER_ID}
        cmake_cxx_compiler_target=${CMAKE_CXX_COMPILER_TARGET}
        cmake_cxx_compiler_version=${CMAKE_CXX_COMPILER_VERSION}
        cmake_cxx_flags=${CMAKE_CXX_FLAGS}
        cmake_cxx_flags_debug=${CMAKE_CXX_FLAGS_DEBUG}
        cmake_cxx_flags_release=${CMAKE_CXX_FLAGS_RELEASE}
        cmake_cxx_flags_relwithdebinfo=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
        cmake_cxx_standard=${CMAKE_CXX_STANDARD}
        cmake_system_name=${CMAKE_SYSTEM_NAME}
        cmake_system_processor=${CMAKE_SYSTEM_PROCESSOR}
        enable_code_timers=${ENABLE_CODE_TIMERS}
        enable_distributed=${ENABLE_DISTRIBUTED}
        tt_enable_light_metal_trace=${TT_ENABLE_LIGHT_METAL_TRACE}
        tt_enable_lto=${TT_ENABLE_LTO}
        tt_metal_skip_linking=${TT_METAL_SKIP_LINKING}
        tt_metal_use_emule=${TT_METAL_USE_EMULE}
        tt_umd_build_simulation=${TT_UMD_BUILD_SIMULATION}
        tt_unity_builds=${TT_UNITY_BUILDS}
        tt_use_system_sfpi=${TT_USE_SYSTEM_SFPI}
)
    list(
        APPEND
        MATMUL_REGISTRY_BUILD_FACT_ARGUMENTS
        --fact
        ${fact}
    )
endforeach()

add_custom_command(
    OUTPUT
        ${MATMUL_REGISTRY_BUILD_HEADER}
    COMMAND
        ${Python3_EXECUTABLE} ${MATMUL_REGISTRY_BUILD_EMITTER} --root ${PROJECT_SOURCE_DIR} --manifest
        ${MATMUL_REGISTRY_SEMANTIC_MANIFEST} --header ${MATMUL_REGISTRY_BUILD_HEADER}
        ${MATMUL_REGISTRY_BUILD_FACT_ARGUMENTS}
    DEPENDS
        ${MATMUL_REGISTRY_BUILD_EMITTER}
        ${MATMUL_REGISTRY_SEMANTIC_MANIFEST}
        ${MATMUL_REGISTRY_SEMANTIC_DEPENDENCIES}
    VERBATIM
)
add_custom_target(matmul_registry_build_attestation DEPENDS ${MATMUL_REGISTRY_BUILD_HEADER})
add_dependencies(ttnn_op_matmul matmul_registry_build_attestation)

# Registry promotion remains an offline release step. Normal TTNN builds never
# regenerate or parse the reviewed table; they only emit private running-build
# digests from the closed semantic manifest and build facts above.
set_source_files_properties(
    ${MATMUL_REGISTRY_GENERATED_SOURCE}
    PROPERTIES
        SKIP_LINTING
            TRUE
        SKIP_UNITY_BUILD_INCLUSION
            ON
)

tt_reuse_precompile_headers(ttnn_op_matmul TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul)
set_target_properties(
    ttnn_op_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
    shared_with_host/*.hpp
)
target_sources(
    ttnn_op_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_MATMUL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_MATMUL_SRCS}
)

target_include_directories(
    ttnn_op_matmul
    PRIVATE
        ${FixmeOpIncDirs}
        ${PROJECT_SOURCE_DIR}/tt_metal
        ${PROJECT_SOURCE_DIR}/tt_metal/api/tt-metalium
        ${PROJECT_SOURCE_DIR}/tt_metal/common
        ${PROJECT_SOURCE_DIR}/tt_metal/hw/inc
        ${MATMUL_REGISTRY_BUILD_DIR}
        ${MATMUL_REGISTRY_GENERATED_DIR}
)
target_link_libraries(ttnn_op_matmul PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_matmul
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/matmul
        COMPONENT ttnn-runtime
)
install(TARGETS ttnn_op_matmul LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_MATMUL_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_MATMUL_NANOBIND_SRCS})
endif()

add_library(ttnn_op_ccl2 ${LIB_TYPE})
add_library(TTNN::Ops::CCL2 ALIAS ttnn_op_ccl2)

target_precompile_headers(ttnn_op_ccl2 REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_ccl2)

set_target_properties(
    ttnn_op_ccl2
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)
# Globbing non-build files is acceptable for now because devs don't generate packages.
file(GLOB_RECURSE kernels all_gather/device/*)
target_sources(
    ttnn_op_ccl2
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES all_gather/all_gather.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        # Common
        ccl2_common.cpp
        # Ops
        all_gather/all_gather.cpp
        all_gather/host/all_gather_program.cpp
)

target_include_directories(ttnn_op_ccl2 PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_ccl2
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_ccl2
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/ccl2
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_ccl2 LIBRARY COMPONENT tar)

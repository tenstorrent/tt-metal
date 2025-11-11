add_library(ttnn_op_experimental_deepseek_b1 ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeepSeekB1 ALIAS ttnn_op_experimental_deepseek_b1)

target_precompile_headers(ttnn_op_experimental_deepseek_b1 REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_b1)

set_target_properties(
    ttnn_op_experimental_deepseek_b1
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# Glob kernels from all operation subfolders
file(GLOB_RECURSE kernels matmul_1d/device/kernels/*)

target_sources(
    ttnn_op_experimental_deepseek_b1
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES matmul_1d/matmul_1d.hpp matmul_1d/device/matmul_1d_device_operation.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        matmul_1d/matmul_1d.cpp
        matmul_1d/device/matmul_1d_device_operation.cpp
        matmul_1d/device/matmul_1d_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_b1 PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_deepseek_b1 PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_experimental_deepseek_b1 LIBRARY COMPONENT tar)

install(
    TARGETS
        ttnn_op_experimental_deepseek_b1
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek_b1
        COMPONENT ttnn-runtime
)

add_library(ttnn_op_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Matmul ALIAS ttnn_op_matmul)

target_precompile_headers(ttnn_op_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_matmul)
set_target_properties(
    ttnn_op_matmul
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

file(GLOB_RECURSE kernels device/kernels/*)
target_sources(
    ttnn_op_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES matmul.hpp device/matmul_op.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/matmul_op.cpp
        device/matmul_op_multi_core_program_factory.cpp
        device/matmul_op_multi_core_reuse_mcast_1d_program_factory.cpp
        device/matmul_op_multi_core_reuse_mcast_2d_program_factory.cpp
        device/matmul_op_multi_core_reuse_mcast_dram_sharded_program_factory.cpp
        device/matmul_op_multi_core_reuse_optimized_program_factory.cpp
        device/matmul_op_multi_core_reuse_program_factory.cpp
        matmul.cpp
)

target_include_directories(ttnn_op_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_matmul
    PRIVATE
        TT::Metalium
        TTNN::Core
)

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

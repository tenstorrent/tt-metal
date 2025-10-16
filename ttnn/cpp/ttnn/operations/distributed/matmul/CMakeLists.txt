add_library(ttnn_op_distributed_matmul ${LIB_TYPE})
add_library(TTNN::Ops::Distributed::Matmul ALIAS ttnn_op_distributed_matmul)

target_precompile_headers(ttnn_op_distributed_matmul REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_distributed_matmul)
set_target_properties(
    ttnn_op_distributed_matmul
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_distributed_matmul
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES distributed_matmul.hpp
    PRIVATE
        distributed_matmul.cpp
)

target_include_directories(ttnn_op_distributed_matmul PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_distributed_matmul
    PUBLIC
        TTNN::Core
        TTNN::Ops::Matmul
    PRIVATE
        TT::Metalium
)

install(TARGETS ttnn_op_distributed_matmul FILE_SET api COMPONENT ttnn-dev)
install(TARGETS ttnn_op_distributed_matmul LIBRARY COMPONENT tar)

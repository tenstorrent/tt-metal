add_library(ttnn_op_experimental_isin ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::IsIn ALIAS ttnn_op_experimental_isin)

target_precompile_headers(ttnn_op_experimental_isin REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_isin)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_isin
    PRIVATE
        device/isin_device_operation.cpp
        device/isin_program_factory.cpp
        isin.cpp
)

target_include_directories(ttnn_op_experimental_isin PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_isin
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_isin LIBRARY COMPONENT tar)

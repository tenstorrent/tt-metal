add_library(ttnn_op_experimental_tosa_scatter ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::TOSAScatter ALIAS ttnn_op_experimental_tosa_scatter)

target_precompile_headers(ttnn_op_experimental_tosa_scatter REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_experimental_tosa_scatter
    PRIVATE
        ../device/scatter_device_operation.cpp
        ../device/scatter_program_factory.cpp
        tosa_scatter.cpp
)

target_include_directories(ttnn_op_experimental_tosa_scatter PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_tosa_scatter
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_tosa_scatter LIBRARY COMPONENT tar)

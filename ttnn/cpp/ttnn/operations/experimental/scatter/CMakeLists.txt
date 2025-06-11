add_library(ttnn_op_experimental_scatter ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Scatter ALIAS ttnn_op_experimental_scatter)

target_precompile_headers(ttnn_op_experimental_scatter REUSE_FROM TT::CommonPCH)

target_sources(
    ttnn_op_experimental_scatter
    PRIVATE
        device/scatter_device_operation.cpp
        device/scatter_program_factory.cpp
        scatter.cpp
)

target_include_directories(ttnn_op_experimental_scatter PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_scatter
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_scatter LIBRARY COMPONENT tar)

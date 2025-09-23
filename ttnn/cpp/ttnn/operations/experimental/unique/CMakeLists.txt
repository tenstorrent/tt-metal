add_library(ttnn_op_experimental_unique ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Unique ALIAS ttnn_op_experimental_unique)

target_precompile_headers(ttnn_op_experimental_unique REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_unique)

target_sources(
    ttnn_op_experimental_unique
    PRIVATE
        device/unique_device_op.cpp
        device/unique_program_factory.cpp
        unique.cpp
)

target_include_directories(ttnn_op_experimental_unique PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_unique
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_unique LIBRARY COMPONENT tar)

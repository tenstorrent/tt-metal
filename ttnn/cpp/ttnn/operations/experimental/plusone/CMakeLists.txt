add_library(ttnn_op_experimental_plusone ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::PlusOne ALIAS ttnn_op_experimental_plusone)

target_precompile_headers(ttnn_op_experimental_plusone REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_plusone)

target_sources(
    ttnn_op_experimental_plusone
    PRIVATE
        device/plusone_op.cpp
        device/plusone_program_factory.cpp
        plusone.cpp
)

target_include_directories(ttnn_op_experimental_plusone PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_plusone
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_experimental_plusone LIBRARY COMPONENT tar)

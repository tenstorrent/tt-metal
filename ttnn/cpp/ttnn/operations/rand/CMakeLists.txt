add_library(ttnn_op_rand ${LIB_TYPE})
add_library(TTNN::Ops::Rand ALIAS ttnn_op_rand)

target_precompile_headers(ttnn_op_rand REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_rand)

target_sources(
    ttnn_op_rand
    PRIVATE
        rand.cpp
        device/rand_device_operation.cpp
        device/rand_program_factory.cpp
)

target_include_directories(ttnn_op_rand PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_rand
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(TARGETS ttnn_op_rand LIBRARY COMPONENT tar)

add_library(ttnn_op_rand ${LIB_TYPE})
add_library(TTNN::Ops::Rand ALIAS ttnn_op_rand)

target_precompile_headers(ttnn_op_rand REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_rand)

target_sources(
    ttnn_op_rand
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES rand.hpp
    PRIVATE
        rand.cpp
        device/rand_device_operation.cpp
        device/rand_program_factory.cpp
)

target_include_directories(ttnn_op_rand PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_rand PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(TARGETS ttnn_op_rand FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

add_library(ttnn_op_experimental_parallel ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::Parallel ALIAS ttnn_op_experimental_parallel)

target_precompile_headers(ttnn_op_experimental_parallel REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_parallel)

set_target_properties(
    ttnn_op_experimental_parallel
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

target_sources(
    ttnn_op_experimental_parallel
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES parallel.hpp
    PRIVATE
        parallel.cpp
        device/parallel_device_operation.cpp
        device/parallel_factory.cpp
)

target_include_directories(ttnn_op_experimental_parallel PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_parallel
    PUBLIC
        TTNN::Core
    PRIVATE
        TT::Metalium
        TTNN::Ops::Normalization
)

install(TARGETS ttnn_op_experimental_parallel LIBRARY COMPONENT tar)

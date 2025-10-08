add_library(ttnn_op_full_like ${LIB_TYPE})
add_library(TTNN::Ops::FullLike ALIAS ttnn_op_full_like)

target_precompile_headers(ttnn_op_full_like REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_full_like)

target_sources(
    ttnn_op_full_like
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES full_like.hpp
    PRIVATE
        device/full_like_device_operation.cpp
        device/full_like_factory.cpp
        full_like.cpp
)

target_include_directories(ttnn_op_full_like PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_full_like PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_full_like
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/full_like
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_full_like LIBRARY COMPONENT tar)

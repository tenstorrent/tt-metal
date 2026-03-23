add_library(ttnn_op_experimental_deltanet ${LIB_TYPE})
add_library(TTNN::Ops::Experimental::DeltaNet ALIAS ttnn_op_experimental_deltanet)

tt_reuse_precompile_headers(ttnn_op_experimental_deltanet TTNN::PCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deltanet)

set_target_properties(
    ttnn_op_experimental_deltanet
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_experimental_deltanet
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES deltanet.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/deltanet_device_operation.cpp
        device/single_core_program_factory.cpp
        deltanet.cpp
)

target_include_directories(ttnn_op_experimental_deltanet PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_experimental_deltanet PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_experimental_deltanet
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deltanet
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deltanet LIBRARY COMPONENT tar)

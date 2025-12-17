add_library(ttnn_op_bernoulli ${LIB_TYPE})
add_library(TTNN::Ops::Bernoulli ALIAS ttnn_op_bernoulli)

target_precompile_headers(ttnn_op_bernoulli REUSE_FROM TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_bernoulli)

set_target_properties(
    ttnn_op_bernoulli
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

file(GLOB_RECURSE kernels device/kernels/*)

target_sources(
    ttnn_op_bernoulli
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        bernoulli.cpp
        device/bernoulli_device_operation.cpp
        device/bernoulli_program_factory.cpp
)

target_include_directories(ttnn_op_bernoulli PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_bernoulli
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_bernoulli
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/bernoulli
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_bernoulli LIBRARY COMPONENT tar)

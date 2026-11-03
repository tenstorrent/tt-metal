include(sources.cmake)

add_library(ttnn_op_toy_scaled_add ${LIB_TYPE})
add_library(TTNN::Ops::ToyScaledAdd ALIAS ttnn_op_toy_scaled_add)

tt_reuse_precompile_headers(ttnn_op_toy_scaled_add TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_toy_scaled_add)

set_target_properties(
    ttnn_op_toy_scaled_add
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The kernels and the args header they share with the program factory.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_toy_scaled_add
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_TOY_SCALED_ADD_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_TOY_SCALED_ADD_SRCS}
)

target_link_libraries(ttnn_op_toy_scaled_add PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_toy_scaled_add
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/toy_scaled_add
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_toy_scaled_add FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target; the
# file list lives in sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_TOY_SCALED_ADD_NANOBIND_SRCS})
endif()

include(sources.cmake)

add_library(ttnn_op_groupnorm_sc_N_1_HW_C_b ${LIB_TYPE})
add_library(TTNN::Ops::GroupnormScN1HwCB ALIAS ttnn_op_groupnorm_sc_N_1_HW_C_b)

tt_reuse_precompile_headers(ttnn_op_groupnorm_sc_N_1_HW_C_b TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_groupnorm_sc_N_1_HW_C_b)

set_target_properties(
    ttnn_op_groupnorm_sc_N_1_HW_C_b
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# The kernels and the args header they share with the program factory.
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
    device/kernels/*.inl
)

target_sources(
    ttnn_op_groupnorm_sc_N_1_HW_C_b
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_GROUPNORM_SC_N_1_HW_C_B_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        ${TTNN_OP_GROUPNORM_SC_N_1_HW_C_B_SRCS}
)

target_link_libraries(ttnn_op_groupnorm_sc_N_1_HW_C_b PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_groupnorm_sc_N_1_HW_C_b
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/groupnorm_sc_N_1_HW_C_b
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_groupnorm_sc_N_1_HW_C_b FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn` Python module target; the
# file list lives in sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on, hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_GROUPNORM_SC_N_1_HW_C_B_NANOBIND_SRCS})
endif()

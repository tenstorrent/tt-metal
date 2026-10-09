include(sources.cmake)

add_library(ttnn_op_bringup_flat_routed_expert_ttnn ${LIB_TYPE})
add_library(TTNN::Ops::Bringup::FlatRoutedExpertTtnn ALIAS ttnn_op_bringup_flat_routed_expert_ttnn)

tt_reuse_precompile_headers(ttnn_op_bringup_flat_routed_expert_ttnn TT::CommonPCH)
# The source sat under ttnn/cpp and got it as an include dir from its API header set's BASE_DIRS.
target_include_directories(ttnn_op_bringup_flat_routed_expert_ttnn PRIVATE ${FixmeOpAPIDir})

set_target_properties(
    ttnn_op_bringup_flat_routed_expert_ttnn
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)

# the kernels and the .hpp beside them (included BY the kernels at JIT compile time) are install artifacts
file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_bringup_flat_routed_expert_ttnn
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${TTNN_OP_BRINGUP_FLAT_ROUTED_EXPERT_TTNN_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        device/flat_routed_expert_device_operation.cpp
        device/flat_routed_expert_plan.cpp
        device/flat_routed_expert_program_factory.cpp
        flat_routed_expert.cpp
)

target_link_libraries(ttnn_op_bringup_flat_routed_expert_ttnn PRIVATE TT::Metalium PUBLIC TTNN::Core)

install(
    TARGETS
        ttnn_op_bringup_flat_routed_expert_ttnn
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/ttnn/bringup/flat_routed_expert_ttnn
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_bringup_flat_routed_expert_ttnn FILE_SET api COMPONENT ttnn-dev LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake. `ttnn` only exists when WITH_PYTHON_BINDINGS is on,
# hence the guard.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_BRINGUP_FLAT_ROUTED_EXPERT_TTNN_NANOBIND_SRCS})
endif()

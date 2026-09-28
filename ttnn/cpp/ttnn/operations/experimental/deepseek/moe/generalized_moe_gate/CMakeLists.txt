# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

include(sources.cmake)

add_library(ttnn_op_experimental_deepseek_moe_generalized_moe_gate ${LIB_TYPE})
add_library(
    TTNN::Ops::Experimental::DeepSeek::MoE::GeneralizedMoeGate
    ALIAS ttnn_op_experimental_deepseek_moe_generalized_moe_gate
)

tt_reuse_precompile_headers(ttnn_op_experimental_deepseek_moe_generalized_moe_gate TT::CommonPCH)
TT_ENABLE_UNITY_BUILD(ttnn_op_experimental_deepseek_moe_generalized_moe_gate)

set_target_properties(
    ttnn_op_experimental_deepseek_moe_generalized_moe_gate
    PROPERTIES
        VERIFY_INTERFACE_HEADER_SETS
            FALSE
)

file(
    GLOB_RECURSE kernels
    device/kernels/*.cpp
    device/kernels/*.hpp
)

target_sources(
    ttnn_op_experimental_deepseek_moe_generalized_moe_gate
    PUBLIC
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES ${kernels}
    PRIVATE
        generalized_moe_gate.cpp
        device/generalized_moe_gate_device_operation.cpp
        device/generalized_moe_gate_program_descriptor_builder.cpp
        device/generalized_moe_gate_program_factory.cpp
)

target_include_directories(ttnn_op_experimental_deepseek_moe_generalized_moe_gate PRIVATE ${FixmeOpIncDirs})
target_link_libraries(
    ttnn_op_experimental_deepseek_moe_generalized_moe_gate
    PRIVATE
        TT::Metalium
        TTNN::Core
)

install(
    TARGETS
        ttnn_op_experimental_deepseek_moe_generalized_moe_gate
    FILE_SET
    kernels
        DESTINATION
            ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/experimental/deepseek/moe/generalized_moe_gate
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_experimental_deepseek_moe_generalized_moe_gate LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's
# own registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_GENERALIZED_MOE_GATE_NANOBIND_SRCS) lives in sources.cmake, not here: this
# file carries metalium-developers-infra as a required co-owner (so infra
# can land sweeping build-system changes), and routine add/remove/rename of
# a nanobind source shouldn't need to go through infra. This wiring line
# itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_EXPERIMENTAL_DEEPSEEK_MOE_GENERALIZED_MOE_GATE_NANOBIND_SRCS})
endif()

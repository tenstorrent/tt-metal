include(sources.cmake)

add_library(ttnn_op_ccl ${LIB_TYPE})
add_library(TTNN::Ops::CCL ALIAS ttnn_op_ccl)

tt_reuse_precompile_headers(ttnn_op_ccl TTNN::PCHFull)
TT_ENABLE_UNITY_BUILD(ttnn_op_ccl)

set_target_properties(
    ttnn_op_ccl
    PROPERTIES
        INTERFACE_HEADER_SETS_TO_VERIFY
            api
)
# Globbing non-build files is acceptable for now because devs don't generate packages.
file(
    GLOB_RECURSE kernels
    all_gather/device/kernels/*.cpp
    all_gather/device/kernels/*.hpp
    broadcast/device/kernels/*.cpp
    common/kernels/*.cpp
    common/kernels/*.hpp
    kernel_common/*.hpp
    kernels/*.cpp
    kernels/*.hpp
    all_to_all_combine/device/kernels/*.cpp
    all_to_all_dispatch/device/kernels/*.cpp
    reduce_to_root/device/kernels/*.cpp
)
target_sources(
    ttnn_op_ccl
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES ${TTNN_OP_CCL_API_HEADERS}
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            # Shared/common headers
            ccl_host_types.hpp
            ccl_nanobind.hpp
            ccl_op_fusion.hpp
            ccl_common.hpp
            ccl_host_datastructures.hpp
            sharding_addrgen_helper.hpp
            shared_with_host/hetergeneous_data_structs.hpp
            shared_with_host/sharded_tensor_addr_gen.hpp
            # common/
            common/host/ccl_command_stream_builders.hpp
            common/host/ccl_worker_builder.hpp
            common/host/command_backend_runtime_args_overrider.hpp
            common/host/moe_utils.hpp
            common/interpreter_backends/kernel_common/algorithms.hpp
            common/interpreter_backends/kernel_common/io_descriptors.hpp
            common/kernels/command_processor.hpp
            common/kernels/ccl_send_reader_two_input.cpp
            common/kernels/ccl_send.cpp
            common/kernels/ccl_wait_completion.cpp
            common/types/ccl_types.hpp
            common/types/ccl_types_args_emitters.hpp
            common/types/ccl_types_device.hpp
            common/types/sharding_common.hpp
            common/uops/ccl_command.hpp
            common/uops/ccl_command_device.hpp
            common/uops/ccl_host_commands.hpp
            common/uops/command_lowering.hpp
            # all_broadcast/
            all_broadcast/all_broadcast.hpp
            all_broadcast/all_broadcast_nanobind.hpp
            all_broadcast/device/all_broadcast_device_operation.hpp
            all_broadcast/device/all_broadcast_device_operation_types.hpp
            all_broadcast/device/all_broadcast_program_factory.hpp
            # all_gather/
            all_gather/all_gather.hpp
            all_gather/all_gather_nanobind.hpp
            all_gather/device/all_gather_device_operation.hpp
            all_gather/device/all_gather_device_operation_types.hpp
            all_gather/device/all_gather_multicast_factory.hpp
            all_gather/device/all_gather_unicast_factory.hpp
            # all_reduce/
            all_reduce/all_reduce.hpp
            all_reduce/all_reduce_nanobind.hpp
            # all_to_all_combine/
            all_to_all_combine/all_to_all_combine.hpp
            all_to_all_combine/all_to_all_combine_nanobind.hpp
            all_to_all_combine/device/all_to_all_combine_device_operation.hpp
            # all_to_all_dispatch/
            all_to_all_dispatch/all_to_all_dispatch.hpp
            all_to_all_dispatch/all_to_all_dispatch_nanobind.hpp
            all_to_all_dispatch/device/all_to_all_dispatch_device_operation.hpp
            # broadcast/
            broadcast/broadcast.hpp
            broadcast/broadcast_nanobind.hpp
            broadcast/device/broadcast_device_operation.hpp
            broadcast/device/broadcast_device_operation_types.hpp
            broadcast/device/broadcast_program_factory.hpp
            # mesh_partition/
            mesh_partition/mesh_partition.hpp
            mesh_partition/mesh_partition_nanobind.hpp
            mesh_partition/device/mesh_partition_device_operation.hpp
            # reduce_scatter/
            reduce_scatter/reduce_scatter.hpp
            reduce_scatter/reduce_scatter_nanobind.hpp
            reduce_scatter/device/reduce_scatter_device_operation.hpp
            # reduce_to_root/
            reduce_to_root/reduce_to_root.hpp
            reduce_to_root/reduce_to_root_nanobind.hpp
            reduce_to_root/device/reduce_to_root_op.hpp
    PRIVATE
        ${TTNN_OP_CCL_SRCS}
)

target_include_directories(ttnn_op_ccl PRIVATE ${FixmeOpIncDirs})
target_link_libraries(ttnn_op_ccl PUBLIC TTNN::Core PRIVATE TT::Metalium)

install(
    TARGETS
        ttnn_op_ccl
    FILE_SET
    api
        COMPONENT ttnn-dev
    FILE_SET
    kernels
        DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/tt-metalium/ttnn/cpp/ttnn/operations/ccl
        COMPONENT ttnn-runtime
)

install(TARGETS ttnn_op_ccl LIBRARY COMPONENT tar)

# Python bindings for this op are registered directly on the shared `ttnn`
# Python module target here, rather than in the centrally-owned
# ttnn/sources.cmake — the target itself is necessarily shared (nanobind
# builds one Python extension module), but which file lists a given op's own
# registration source doesn't need to be. `ttnn` only exists when
# WITH_PYTHON_BINDINGS is on, hence the guard.
#
# The actual file list (TTNN_OP_CCL_NANOBIND_SRCS) lives in sources.cmake,
# not here: this file carries metalium-developers-infra as a required
# co-owner (so infra can land sweeping build-system changes), and routine
# add/remove/rename of a nanobind source shouldn't need to go through infra.
# This wiring line itself is build logic and is expected to change rarely.
if(TARGET ttnn)
    target_sources(ttnn PRIVATE ${TTNN_OP_CCL_NANOBIND_SRCS})
endif()

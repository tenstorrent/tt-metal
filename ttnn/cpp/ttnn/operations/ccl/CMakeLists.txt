include(sources.cmake)

add_library(ttnn_op_ccl ${LIB_TYPE})
add_library(TTNN::Ops::CCL ALIAS ttnn_op_ccl)

tt_reuse_precompile_headers(ttnn_op_ccl TTNN::PCH)
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
    broadcast/device/kernels/*
    common/kernels/*
    kernel_common/*
    kernels/*
    all_to_all_combine/device/kernels/*
    all_to_all_dispatch/device/kernels/*
    reduce_to_root/device/kernels/*
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

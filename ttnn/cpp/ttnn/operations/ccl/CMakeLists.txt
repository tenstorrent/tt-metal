add_library(ttnn_op_ccl ${LIB_TYPE})
add_library(TTNN::Ops::CCL ALIAS ttnn_op_ccl)

target_precompile_headers(ttnn_op_ccl REUSE_FROM TT::CommonPCH)
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
    common/kernels/*
    kernel_common/*
    kernels/*
    all_to_all_combine/device/kernels/*
    all_to_all_dispatch/device/kernels/*
)
target_sources(
    ttnn_op_ccl
    PUBLIC
        FILE_SET api
        TYPE HEADERS
        BASE_DIRS ${FixmeOpAPIDir}
        FILES
            all_gather/all_gather.hpp
            all_reduce/all_reduce.hpp
            reduce_scatter/reduce_scatter.hpp
            all_broadcast/device/all_broadcast_device_operation_types.hpp
            all_broadcast/device/all_broadcast_device_operation.hpp # seems odd to have some of this file, but it's needed for tt-train to consume
            all_broadcast/device/all_broadcast_program_factory.hpp
            ccl_common.hpp
            ccl_host_datastructures.hpp
            ccl_host_types.hpp
            ccl_op_fusion.hpp
            common/host/ccl_command_stream_builders.hpp
            common/host/moe_utils.hpp
            common/types/ccl_types.hpp
            common/uops/ccl_command.hpp
            mesh_partition/mesh_partition.hpp
            shared_with_host/hetergeneous_data_structs.hpp
        FILE_SET kernels
        TYPE HEADERS
        BASE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${kernels}
            ccl_host_types.hpp
            shared_with_host/hetergeneous_data_structs.hpp
            shared_with_host/sharded_tensor_addr_gen.hpp
            common/types/ccl_types.hpp
            common/types/sharding_common.hpp
            common/types/ccl_types_device.hpp
            common/uops/ccl_command_device.hpp
            common/uops/ccl_command.hpp
            common/interpreter_backends/kernel_common/algorithms.hpp
            common/interpreter_backends/kernel_common/io_descriptors.hpp
            common/kernels/command_processor.hpp
            common/kernels/ccl_send_reader_two_input.cpp
            common/kernels/ccl_send.cpp
            common/kernels/ccl_wait_completion.cpp
    PRIVATE
        # Common
        ccl_op_fusion.cpp
        ccl_common.cpp
        ccl_host_datastructures.cpp
        common/types/ccl_types_args_emitters.cpp
        common/host/command_backend_runtime_args_overrider.cpp
        common/uops/ccl_command.cpp
        common/uops/command_lowering.cpp
        common/uops/ccl_host_commands.cpp
        common/host/ccl_worker_builder.cpp
        common/host/ccl_command_stream_builders.cpp
        common/host/moe_utils.cpp
        # Ops
        all_broadcast/all_broadcast.cpp
        all_broadcast/device/all_broadcast_device_operation.cpp
        all_broadcast/device/all_broadcast_program_factory.cpp
        all_gather/all_gather.cpp
        all_gather/device/all_gather_device_operation.cpp
        all_gather/device/all_gather_program_factory.cpp
        all_reduce/all_reduce.cpp
        all_to_all_combine/all_to_all_combine.cpp
        all_to_all_combine/device/all_to_all_combine_device_operation.cpp
        all_to_all_combine/device/all_to_all_combine_program_factory.cpp
        all_to_all_dispatch/all_to_all_dispatch.cpp
        all_to_all_dispatch/device/all_to_all_dispatch_device_operation.cpp
        all_to_all_dispatch/device/all_to_all_dispatch_program_factory.cpp
        reduce_to_root/reduce_to_root.cpp
        reduce_to_root/device/reduce_to_root_op.cpp
        reduce_to_root/device/reduce_to_root_program.cpp
        reduce_scatter/reduce_scatter.cpp
        reduce_scatter/device/reduce_scatter_device_operation.cpp
        reduce_scatter/device/reduce_scatter_program_factory.cpp
        mesh_partition/mesh_partition.cpp
        mesh_partition/device/mesh_partition_device_operation.cpp
        mesh_partition/device/mesh_partition_program_factory.cpp
        broadcast/broadcast.cpp
        broadcast/device/broadcast_op.cpp
        broadcast/device/broadcast_program.cpp
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

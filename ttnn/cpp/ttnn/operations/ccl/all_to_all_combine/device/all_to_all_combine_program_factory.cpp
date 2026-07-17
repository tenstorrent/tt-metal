// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/all_to_all_combine/device/all_to_all_combine_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

namespace {

// Build the ProgramDescriptor for one device's slice of the all-to-all-combine
// workload.  Per-coord variation is real here: src_chip_id, the linearized mesh
// index, the DIRECTIONS define (writer kernel) and the fabric-connection runtime
// args all depend on `mesh_coordinate`, so callers must build one descriptor
// per coord and cannot reuse a single descriptor across the mesh.
tt::tt_metal::ProgramDescriptor build_combine_program_descriptor(
    const AllToAllCombineDeviceOperation::operation_attributes_t& operation_attributes,
    const AllToAllCombineDeviceOperation::tensor_args_t& tensor_args,
    AllToAllCombineDeviceOperation::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tt::tt_metal::GlobalSemaphore& init_semaphore,
    const tt::tt_metal::GlobalSemaphore& cross_device_semaphore) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    ProgramDescriptor desc;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& output_tensor = tensor_return_value;
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    const auto input_dtype = input_tensor.dtype();

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const auto& mapping_shape = mapping_tensor.tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();
    const uint32_t batch_size = metadata_shape[1];
    const uint32_t seq_size = metadata_shape[2];
    const uint32_t selected_experts_k = metadata_shape[-1];
    const uint32_t experts = mapping_shape[-2];

    TT_FATAL(experts % num_devices == 0, "Currently assuming that experts are evenly split among devices");
    const uint32_t experts_per_device = experts / num_devices;

    const auto& input_spec = input_tensor.tensor_spec();
    const auto& mapping_spec = mapping_tensor.tensor_spec();
    const auto& metadata_spec = metadata_tensor.tensor_spec();

    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;

    const auto input_page_size_bytes = input_spec.compute_page_size_bytes();
    const auto mapping_page_size_bytes = mapping_spec.compute_page_size_bytes();
    const auto metadata_page_size_bytes = metadata_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto dram_alignment = hal::get_dram_alignment();

    const auto aligned_input_page_size_bytes = tt::align(input_page_size_bytes, input_is_dram? dram_alignment:l1_alignment);
    const auto aligned_mapping_page_size_bytes = tt::align(mapping_page_size_bytes, l1_alignment);
    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto mapping_data_format = datatype_to_dataformat_converter(mapping_tensor.dtype());
    const auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.dtype());

    // Anything less will lead to deadlocks. It's clear why, TODO fix it.
    const uint32_t buffering_factor = experts_per_device;

    const auto subdevice_cores = corerange_to_cores(operation_attributes.worker_core_range_set);

    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    // perform work-split across cores
    uint32_t tokens_per_device = batch_size * seq_size;
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(tokens_per_device, tokens_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, operation_attributes.worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // input sharded buffer
    constexpr auto data_cb_id = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffering_factor * aligned_input_page_size_bytes,
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(data_cb_id),
            .data_format = input_data_format,
            .page_size = aligned_input_page_size_bytes,
        }}},
    });

    // full mapping buffer
    constexpr auto mapping_tensor_cb_id = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_mapping_page_size_bytes,
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(mapping_tensor_cb_id),
            .data_format = mapping_data_format,
            .page_size = aligned_mapping_page_size_bytes,
        }}},
    });

    // scratch space to store and share indices of per device experts
    constexpr auto local_experts_cb_id = tt::CBIndex::c_2;
    using local_experts_t = uint16_t;
    const auto aligned_local_expert_page_size_bytes =
        tt::align(experts_per_device * sizeof(local_experts_t), l1_alignment);
    const auto local_experts_dataformat = datatype_to_dataformat_converter(convert_to_data_type<local_experts_t>());
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_local_expert_page_size_bytes,
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(local_experts_cb_id),
            .data_format = local_experts_dataformat,
            .page_size = aligned_local_expert_page_size_bytes,
        }}},
    });

    // metadata page buffer
    constexpr auto metadata_cb_id = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_metadata_page_size_bytes,
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(metadata_cb_id),
            .data_format = metadata_data_format,
            .page_size = aligned_metadata_page_size_bytes,
        }}},
    });

    // client interface
    // [0] = data unicast header; [1]/[2] = completion atomic-inc bidirectional multicast headers
    // (positive/negative ring directions); [1] is reused earlier for the init-semaphore send.
    constexpr auto num_headers = 3;
    constexpr auto client_interface_cb_id = tt::CBIndex::c_4;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_headers * CLIENT_INTERFACE_SIZE,
        .core_ranges = sender_core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(client_interface_cb_id),
            .data_format = tt::DataFormat::UInt32,
            .page_size = CLIENT_INTERFACE_SIZE,
        }}},
    });

    const uint32_t flat_mesh_idx = common::get_linearized_index(mesh_coordinate, mesh_view);

    std::vector<uint32_t> reader_compile_time_args = {
        mapping_tensor_cb_id,
        local_experts_cb_id,
        metadata_cb_id,
        data_cb_id,
        experts_per_device,
        batch_size,
        seq_size,
        experts,  // same as num_mapping_pages
        flat_mesh_idx,
        input_page_size_bytes,
        selected_experts_k,
        mapping_page_size_bytes,
        metadata_page_size_bytes,
        operation_attributes.locally_reduced,
    };
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(mapping_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_compile_time_args);

    const auto& axis = operation_attributes.axis;

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    std::vector<uint32_t> writer_compile_time_args = {
        metadata_cb_id,
        local_experts_cb_id,
        client_interface_cb_id,
        data_cb_id,
        batch_size,
        seq_size,
        selected_experts_k,
        experts_per_device,
        num_devices,
        src_chip_id,
        input_page_size_bytes,
        l1_alignment,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        max_packet_size_bytes,
        common::get_linearized_index(mesh_coordinate, mesh_view),
        (uint32_t)topology,
        operation_attributes.locally_reduced,
    };
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    // fabric routing info — enumerated for every device in the mesh in row-major order to
    // build the DEST_CHIP_ID / DEST_MESH_ID define arrays consumed by the writer kernel.
    // Mirrors the legacy use of `all_mesh_coordinates` (passed as tensor_coords.coords()),
    // which for these CCL ops covers the entire mesh.
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord_fabric_node_id : mesh_view.get_fabric_node_ids()) {
        dest_mesh_id.push_back(*coord_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)coord_fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] = common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
        {"DIRECTIONS", common::stringify(directions)}};

    if (axis.has_value()) {
        writer_defines["REPLICATE_GROUP_AXIS"] = std::to_string(axis.value());
    }

    // Build kernel descriptors.  Push them onto desc.kernels NOW (before the per-link
    // runtime-args loop) so we can refer to them by stable index for both emplace_runtime_args
    // and the fabric helper, which expects a Program/Descriptor reference but does not need
    // a KernelHandle for this call site.
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_core_grid;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
    };

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/writer_all_to_all_combine.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_core_grid;
    writer_kernel_desc.compile_time_args = writer_compile_time_args;
    writer_kernel_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
    };

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    constexpr KernelHandle ternary_reader_kernel_id = 0;
    constexpr KernelHandle unary_writer_kernel_id = 1;

    uint32_t link_id = 0;
    uint32_t tokens_per_core_start = 0;
    log_debug(tt::LogOp, "Runtime arguments are being calculated for MeshCoordinate {}", mesh_coordinate);
    for (const auto& sender_core : sender_cores) {
        // Reader runtime args: indices 0/1/2 are Buffer*'s for BufferBinding fast-path.
        const uint32_t token_range_start = tokens_per_core_start;
        const uint32_t token_range_end = std::min(tokens_per_core_start + tokens_per_core, tokens_per_device);
        tokens_per_core_start = token_range_end;

        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(mapping_tensor.buffer());
        reader_rt_args.push_back(metadata_tensor.buffer());
        reader_rt_args.push_back(input_tensor.buffer());
        reader_rt_args.push_back(token_range_start);
        reader_rt_args.push_back(token_range_end);
        desc.kernels[ternary_reader_kernel_id].emplace_runtime_args(sender_core, reader_rt_args);

        // The fabric helper appends to a plain std::vector<uint32_t>; build the writer
        // args there first, then promote them onto the kernel descriptor.  Index 0 is
        // the output buffer's base address — push it as Buffer* so the framework records
        // a BufferBinding for the cache-hit fast path.  All other positions remain
        // plain uint32_t (semaphores stable across cache hits).
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // placeholder; replaced via Buffer* below
            (uint32_t)cross_device_semaphore.address(),
            (uint32_t)init_semaphore.address(),
            token_range_start,
            token_range_end,
        };
        log_debug(
            tt::LogOp,
            "Setting runtime args for core {}. It will operate on tokens {} to {}. Global semaphore address: {}",
            sender_core,
            token_range_start,
            token_range_end,
            (uint32_t)cross_device_semaphore.address());

        for (const auto& neighbor_coordinate : neighbors) {
            const auto neighbor_fabric_id = mesh_device->get_fabric_node_id(neighbor_coordinate);
            append_fabric_connection_rt_args<ProgramDescriptor>(
                fabric_node_id, neighbor_fabric_id, link_id, desc, sender_core, writer_runtime_args);
        }

        KernelDescriptor::RTArgList writer_rt_args_builder;
        writer_rt_args_builder.reserve(writer_runtime_args.size());
        writer_rt_args_builder.push_back(output_tensor.buffer());
        for (size_t i = 1; i < writer_runtime_args.size(); ++i) {
            writer_rt_args_builder.push_back(writer_runtime_args[i]);
        }
        desc.kernels[unary_writer_kernel_id].emplace_runtime_args(sender_core, writer_rt_args_builder);
        link_id++;
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt::tt_metal;

    // Workload-scoped resources: allocate the two GlobalSemaphores once per
    // cache miss and run the cross-device Synchronize barrier here so it's
    // amortised across every per-coord program build below.  Storing them on
    // WorkloadDescriptor::semaphores hands ownership to the program-cache so
    // they stay alive for the lifetime of the cached MeshWorkload.
    auto* mesh_device = tensor_args.input_tensor.device();
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    WorkloadDescriptor workload_descriptor;
    workload_descriptor.semaphores.push_back(init_barrier_semaphore);
    workload_descriptor.semaphores.push_back(final_barrier_semaphore);

    // Per-coord variation (src_chip_id, linearized mesh index, fabric-connection
    // runtime args, DIRECTIONS define) means we MUST build one ProgramDescriptor
    // per coord rather than reuse a single descriptor across the mesh.
    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        ProgramDescriptor desc = build_combine_program_descriptor(
            operation_attributes,
            tensor_args,
            tensor_return_value,
            coord,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

}  // namespace ttnn::operations::ccl

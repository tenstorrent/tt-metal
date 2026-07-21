// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
///

#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/global_semaphore.hpp"

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <algorithm>

namespace ttnn::prim {

namespace {

// Build a per-coord ProgramDescriptor.  The kernel args depend on the sender
// coordinate (ring_index + forward/backward fabric neighbor lookups), so this
// runs once per coord inside create_workload_descriptor().
tt::tt_metal::ProgramDescriptor build_program_descriptor_at(
    const AllBroadcastParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const Tensor& input,
    std::vector<Tensor>& output_tensors,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = input;
    tt::tt_metal::ProgramDescriptor desc;

    auto* mesh_device = input_tensor.device();

    uint32_t ring_size = operation_attributes.ring_size;
    uint32_t ring_index = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    bool tilized = input_tensor.layout() == ttnn::TILE_LAYOUT;

    uint32_t num_width_shards = 1;
    if (!tilized && (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                     input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED)) {
        num_width_shards = input_tensor.padded_shape()[-1] / input_tensor.memory_config().shard_spec()->shape[1];
    }
    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
        ring_size, ring_index, operation_attributes.topology, true);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] = ::ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        num_workers_per_link,
        mesh_device,
        operation_attributes.sub_device_id,
        CoreCoord(0, 0),
        std::nullopt);

    // Info for RM tensors
    uint32_t row_size = input_tensor.logical_shape()[-1] * input_tensor.element_size();
    uint32_t page_size = input_tensor.buffer()->aligned_page_size();

    uint32_t num_rows = std::accumulate(
        input_tensor.logical_shape().cbegin(),
        input_tensor.logical_shape().cend() - 1,
        1u,
        std::multiplies<uint32_t>());

    // L1 Scratch CB Creation
    DataType dtype = input_tensor.dtype();
    const uint32_t fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t MAX_PACKET_SIZE_BYTES =
        dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;
    const size_t packet_size_bytes =
        tilized ? tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes() : MAX_PACKET_SIZE_BYTES;
    size_t max_packet_size = packet_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t cb_total_size = cb_num_pages * l1_scratch_cb_page_size_bytes;
    uint32_t cb_page_size = l1_scratch_cb_page_size_bytes;

    uint32_t buffer_page_size = page_size;
    uint32_t num_packets_per_page =
        static_cast<uint32_t>(std::ceil(static_cast<double>(buffer_page_size) / max_packet_size));
    if (!tilized) {
        if (num_width_shards > 1) {
            buffer_page_size = input_tensor.memory_config().shard_spec()->shape[1] * input_tensor.element_size();
        }

        uint32_t num_rows_per_packet = (max_packet_size / buffer_page_size >= 2) ? 2 : 1;
        cb_total_size = 3 * buffer_page_size * num_rows_per_packet;
        cb_page_size = buffer_page_size;
    }
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_total_size,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = cb_page_size,
        }}},
    });

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,                               // cb0_id
        num_pages_per_packet,                        // packet_size_in_pages
        input_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
        true,                                        // is_sender
    };

    if (!tilized) {
        reader_compile_args = {
            src0_cb_index,                                      // cb0_id
            buffer_page_size,                                   // page_size
            row_size,                                           // row_size
            (max_packet_size / buffer_page_size >= 2) ? 2 : 1,  // num_rows_per_packet
            num_packets_per_page,                               // num_packets_per_page
            max_packet_size,
            true,  // is_sender
        };
    }

    // Writer kernel
    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,                               // cb0_id
        num_pages_per_packet,                        // packet_size_in_pages
        input_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
        num_targets_forward,                         // num_targets_forward_direction
        num_targets_backward,                        // num_targets_backward_direction
        true,                                        // is_sender
    };

    if (!tilized) {
        writer_compile_args = {
            src0_cb_index,  // cb0_id
            buffer_page_size,
            row_size,
            max_packet_size,
            (max_packet_size / buffer_page_size >= 2) ? 2 : 1,  // num_rows_per_packet
            num_packets_per_page,                               // num_packets_per_row
            num_targets_forward,                                // num_targets_forward_direction
            num_targets_backward,                               // num_targets_backward_direction
            true,                                               // is_sender
        };
    }
    std::vector<uint32_t> mcast_forward_args(2, 0);
    std::vector<uint32_t> mcast_backward_args(2, 0);
    if (forward_coord.has_value()) {
        mcast_forward_args[0] = 1;
        mcast_forward_args[1] = num_targets_forward;
    }
    if (backward_coord.has_value()) {
        mcast_backward_args[0] = 1;
        mcast_backward_args[1] = num_targets_backward;
    }
    writer_compile_args.insert(writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input_tensor, reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(input_tensor, writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(writer_compile_args);
    }

    // Build kernel descriptors.  Push them onto desc.kernels NOW (before the
    // per-link runtime-args loop) so we can refer to them by stable index for
    // both emplace_runtime_args() and the fabric helper, which expects a
    // KernelHandle that indexes into desc.kernels for the ProgramDescriptor
    // overload.
    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        tilized ? "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_reader.cpp"
                : "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.defines = {kernel_defines.begin(), kernel_defines.end()};
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        tilized ? "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_tile_writer.cpp"
                : "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/broadcast_rm_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.defines = {kernel_defines.begin(), kernel_defines.end()};
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    tt::tt_metal::KernelHandle worker_sender_reader_kernel_id = 0;
    tt::tt_metal::KernelHandle worker_sender_writer_kernel_id = 1;

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / operation_attributes.num_links;
        if (!tilized) {
            base_pages_per_worker = num_rows / operation_attributes.num_links;
        }
        uint32_t remainder = input_tensor_num_pages % operation_attributes.num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

        // Build reader RT args.  The first arg is the input tensor buffer
        // address; push it as Buffer* so the framework records a BufferBinding
        // and patches it directly on cache hit (no override_runtime_arguments).
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_tensor.buffer());                   // tensor_address0
        reader_rt_args.push_back(input_tile_id_start * num_width_shards);  // tile_id_start
        reader_rt_args.push_back(input_tile_id_end * num_width_shards);    // tile_id_end

        if (sharded) {
            // shard_builder still writes raw uint32_t into a std::vector; append
            // those onto the RTArgList builder.
            std::vector<uint32_t> sharding_rt_args;
            shard_builder::extend_sharding_run_time_args(input_tensor, sharding_rt_args);
            reader_rt_args.append(sharding_rt_args);
        }
        desc.kernels[worker_sender_reader_kernel_id].emplace_runtime_args(core, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = ring_size * operation_attributes.num_links;
        uint32_t output_tile_id_start = input_tile_id_start;
        uint32_t output_tile_id_end = input_tile_id_end;

        // The fabric helper appends to a plain std::vector<uint32_t>; build the
        // writer args there first, then push them into the kernel descriptor.
        // Index 0 is a placeholder (output buffer base address) that is replaced
        // with a Buffer* binding before emplace_runtime_args() below.
        std::vector<uint32_t> writer_rt_args = {
            output_tensors[ring_index]
                .buffer()
                ->address(),                          // tensor_address0 (placeholder; replaced via Buffer* below)
            semaphore.address(),                      // out_ready_sem_bank_addr (absolute address)
            output_tile_id_start * num_width_shards,  // tile_id_start
            output_tile_id_end * num_width_shards,    // tile_id_end
            wait_output_semaphore,                    // wait_output_semaphore
            reset_global_semaphore,                   // reset_global_semaphore
            drain_sync_core.x,                        // out_ready_sem_noc0_x
            drain_sync_core.y,                        // out_ready_sem_noc0_y
            out_ready_sem_wait_value,                 // out_ready_sem_wait_value
            barrier_semaphore.address(),              // barrier_sem
            barrier_core.x,                           // barrier_sem_noc0_x
            barrier_core.y                            // barrier_sem_noc0_y
        };
        auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
        writer_rt_args.push_back(num_connections);
        if (sharded) {
            shard_builder::extend_sharding_run_time_args(input_tensor, writer_rt_args);
        }

        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
        dst_nodes.reserve(num_connections);
        if (forward_coord.has_value()) {
            const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            dst_nodes.push_back(forward_coord_fabric_node_id);
        }
        if (backward_coord.has_value()) {
            const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            dst_nodes.push_back(backward_coord_fabric_node_id);
        }

        // The fabric helper's ProgramDescriptor specialization indexes into
        // desc.kernels via the KernelHandle to add per-kernel defines.  It also
        // appends additional fabric connection args (and, on 2D meshes, more
        // header words) onto writer_rt_args.
        append_routing_plane_connection_manager_rt_args(
            sender_fabric_node_id, dst_nodes, {link}, desc, worker_sender_writer_kernel_id, core, writer_rt_args);

        // Promote the writer RT args to the kernel descriptor.  Index 0 is the
        // output buffer's base address — push it as Buffer* so the framework
        // records a BufferBinding for the cache-hit fast path.  All other
        // positions remain plain uint32_t.
        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_args_builder;
        writer_rt_args_builder.reserve(writer_rt_args.size());
        writer_rt_args_builder.push_back(output_tensors[ring_index].buffer());
        for (size_t i = 1; i < writer_rt_args.size(); ++i) {
            writer_rt_args_builder.push_back(writer_rt_args[i]);
        }
        desc.kernels[worker_sender_writer_kernel_id].emplace_runtime_args(core, writer_rt_args_builder);
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllBroadcastProgramFactory::create_workload_descriptor(
    const AllBroadcastParams& operation_attributes,
    const Tensor& input,
    std::vector<Tensor>& output_tensors,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = input.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Allocate the two workload-scoped GlobalSemaphores.  Park them on the
    // descriptor's `semaphores` vector so their device-side allocations stay
    // alive for the lifetime of the cached workload — both are referenced by
    // writer runtime args as absolute addresses.  The init barrier is stored
    // at index 0 and the out-ready drain semaphore at index 1; subscript
    // access below mirrors the original argument ordering at the create_at
    // call site (semaphore = final drain, barrier_semaphore = init barrier).
    auto sem_buffer_type = operation_attributes.use_l1_small_for_semaphores ? tt::tt_metal::BufferType::L1_SMALL
                                                                            : tt::tt_metal::BufferType::L1;
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    workload_descriptor.semaphores.push_back(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type));
    const auto& init_barrier_semaphore = workload_descriptor.semaphores[0];
    const auto& final_barrier_semaphore = workload_descriptor.semaphores[1];
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    // Build a per-coord ProgramDescriptor.  Unlike pool/upsample, all_broadcast's
    // per-coord program depends on the sender coordinate (ring_index +
    // forward/backward fabric neighbors), so we cannot share one descriptor
    // across the mesh — emit one entry per coord.
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_program_descriptor_at(
            operation_attributes, coord, input, output_tensors, final_barrier_semaphore, init_barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::prim

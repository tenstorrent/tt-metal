// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_via_broadcast_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <bit>
#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <variant>

namespace ttnn {

using namespace ccl;

namespace experimental::prim {

namespace {

// Worker-core selection close to the ethernet routers.  Preserves the
// hand-tuned core layout from the legacy factory (ethernet locality matters
// for the broadcast all-gather's fabric throughput).
CoreRangeSet get_cores_close_to_erisc(uint32_t num_workers, bool row_wise) {
    CoreRangeSet worker_cores;
    std::vector<CoreRange> desired_core_range = {CoreRange({5, 3}, {6, 3}), CoreRange({2, 8}, {3, 8})};
    for (const auto& cr : desired_core_range) {
        auto cores = corerange_to_cores(cr, std::nullopt, row_wise);
        for (const auto& core : cores) {
            worker_cores = worker_cores.merge(CoreRangeSet(CoreRange(core, core)));
            if (worker_cores.num_cores() == num_workers) {
                break;
            }
        }
        if (worker_cores.num_cores() == num_workers) {
            break;
        }
    }
    return worker_cores;
}

// Build a ProgramDescriptor for one mesh coord.  The per-coord layout is
// otherwise identical to the legacy factory's create_at(); the only changes
// are:
//   - kernels/CBs are pushed onto a ProgramDescriptor rather than created
//     directly on a Program,
//   - the writer kernel index (used as KernelHandle for the templated fabric
//     helper) is the descriptor index of the writer KernelDescriptor.
tt::tt_metal::ProgramDescriptor build_descriptor_at(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const Tensor& input,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = input;
    tt::tt_metal::ProgramDescriptor desc;

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

    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
        ring_size, ring_index, operation_attributes.topology, true);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;

    auto sender_worker_core_range =
        get_cores_close_to_erisc(operation_attributes.num_links * num_workers_per_link, true);
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range);

    const bool row_major = input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT;
    const bool nd_sharded = input_tensor.memory_config().created_with_nd_shard_spec();
    const uint32_t MAX_PACKET_SIZE_BYTES = std::bit_floor(tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes());
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();
    uint32_t rm_rows_per_packet = 1;
    uint32_t cb_page_size = 0;

    if (row_major) {
        TT_FATAL(
            input_page_size == output_page_size,
            "Row-major broadcast all-gather requires equal input/output page sizes, got {} and {}",
            input_page_size,
            output_page_size);
        rm_rows_per_packet = (MAX_PACKET_SIZE_BYTES / input_page_size >= 2) ? 2 : 1;
        if (nd_sharded) {
            rm_rows_per_packet = std::max<uint32_t>(1, std::min<uint32_t>(MAX_PACKET_SIZE_BYTES / input_page_size, 8));
        }
        cb_page_size = input_page_size * rm_rows_per_packet;
    } else {
        cb_page_size = std::lcm(std::lcm(input_page_size, output_page_size), MAX_PACKET_SIZE_BYTES);
        // 32^2 elements == 1/2 or 1 packet, a couple more packets per cb_page for less sync
        cb_page_size *= 4;
    }

    const auto& output_shape = output_tensor.logical_shape();
    const uint32_t local_height = output_shape[operation_attributes.dim] / ring_size;
    const uint32_t gathered_height = output_shape[operation_attributes.dim];
    // Row-major selection produces a compact per-sender row range. Tiled input retains the
    // buffer-derived page count because logical dimensions do not describe tile pages directly.
    const uint32_t num_input_pages =
        row_major ? std::accumulate(output_shape.cbegin(), output_shape.cend() - 1, 1u, std::multiplies<uint32_t>()) /
                        ring_size
                  : input_tensor.buffer()->num_pages();
    TT_FATAL(num_input_pages > 0, "Broadcast all-gather requires at least one input page");
    TT_FATAL(
        (num_input_pages * input_page_size) % output_page_size == 0,
        "Broadcast all-gather requires per-device bytes ({}) to be divisible by output page size ({})",
        num_input_pages * input_page_size,
        output_page_size);
    const uint32_t num_output_pages = (num_input_pages * input_page_size) / output_page_size;
    // Tiled output uses a contiguous sender slab. Row-major output rows are mapped in the writer
    // because batch/head planes introduce gaps between sender slabs.
    uint32_t write_page_offset = row_major ? 0 : num_output_pages * ring_index;

    // L1 Scratch CB Creation
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 3 * cb_page_size,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = row_major ? input_page_size : cb_page_size,
        }}},
    });

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args;
    std::map<std::string, std::string> reader_defines;
    if (row_major) {
        reader_compile_args = {
            src0_cb_index,
            input_page_size,
            input_tensor.logical_shape()[-1] * input_tensor.element_size(),
            rm_rows_per_packet,
            static_cast<uint32_t>(tt::div_up(input_page_size, MAX_PACKET_SIZE_BYTES)),
            MAX_PACKET_SIZE_BYTES,
            true,
        };
        if (operation_attributes.batch_slice_idx.has_value() || operation_attributes.valid_gather_extent.has_value()) {
            reader_defines["SELECT_INPUT_ROWS"] = "1";
        }
        if (nd_sharded) {
            const auto& input_shape = input_tensor.logical_shape();
            const auto& shard_shape = input_tensor.nd_shard_spec()->shard_shape;
            const bool contiguous_row_blocks =
                input_shape.rank() == 4 && shard_shape.rank() == 4 && input_shape[1] == 1 && shard_shape[0] == 1 &&
                shard_shape[1] == 1 && shard_shape[-1] == input_shape[-1] && input_shape[-2] % shard_shape[-2] == 0;
            if (contiguous_row_blocks) {
                reader_defines["ND_FULL_WIDTH_ROW_BLOCKS"] = "1";
                reader_defines["ND_ROWS_PER_SHARD"] = std::to_string(shard_shape[-2]);
            }
        }
    } else {
        reader_compile_args = {src0_cb_index, input_page_size, cb_page_size};
    }

    // Writer kernel
    std::vector<uint32_t> writer_compile_args;
    std::map<std::string, std::string> writer_defines;
    if (row_major) {
        writer_compile_args = {
            src0_cb_index,
            output_page_size,
            input_tensor.logical_shape()[-1] * input_tensor.element_size(),
            MAX_PACKET_SIZE_BYTES,
            rm_rows_per_packet,
            static_cast<uint32_t>(tt::div_up(output_page_size, MAX_PACKET_SIZE_BYTES)),
            num_targets_forward,
            num_targets_backward,
            true,
        };
        writer_defines["DIRECT_GATHER_OUTPUT_ROWS"] = "1";
    } else {
        writer_compile_args = {
            src0_cb_index,
            cb_page_size,
            output_page_size,
            MAX_PACKET_SIZE_BYTES,
            num_targets_forward,
            num_targets_backward,
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
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    // Kernels are pushed in a fixed order — reader first, then writer.  The
    // writer's descriptor index becomes its KernelHandle, which the templated
    // fabric helper uses to add defines.
    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source = row_major ? "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/"
                                                   "broadcast_rm_reader.cpp"
                                                 : "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/"
                                                   "kernels/broadcast_rm_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.defines = {reader_defines.begin(), reader_defines.end()};
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};
    const tt::tt_metal::KernelHandle reader_kernel_index = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(reader_kernel_desc));

    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source = row_major ? "ttnn/cpp/ttnn/operations/ccl/broadcast/device/kernels/"
                                                   "broadcast_rm_writer.cpp"
                                                 : "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/"
                                                   "kernels/broadcast_rm_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.defines = {writer_defines.begin(), writer_defines.end()};
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};
    tt::tt_metal::KernelHandle writer_kernel_index = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kernel_desc));

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto* mesh_device = input_tensor.device();
    CoreCoord barrier_core;
    const auto& input_shape = input_tensor.logical_shape();
    uint32_t middle_volume = 1;
    for (uint32_t dim_index = 1; dim_index + 2 < input_shape.rank(); ++dim_index) {
        middle_volume *= input_shape[dim_index];
    }
    const uint32_t input_height = input_shape[-2];
    const uint32_t input_rows_per_batch = middle_volume * input_height;
    const uint32_t output_rows_per_batch = middle_volume * local_height;
    const uint32_t batch_begin = operation_attributes.batch_slice_idx.value_or(0);
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set reader runtime args
        uint32_t input_pages_per_link = num_input_pages / operation_attributes.num_links;
        uint32_t remainder = num_input_pages % operation_attributes.num_links;
        uint32_t input_tile_id_start = (link * input_pages_per_link) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);

        // tensor_address0 is bound as a Buffer*; the framework patches it on
        // cache hits via the buffer_bindings fast path.
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_tensor.buffer());
        reader_rt_args.push_back(input_tile_id_start);
        reader_rt_args.push_back(input_tile_id_end);
        if (row_major) {
            reader_rt_args.push_back(input_rows_per_batch);
            reader_rt_args.push_back(output_rows_per_batch);
            reader_rt_args.push_back(input_height);
            reader_rt_args.push_back(local_height);
            reader_rt_args.push_back(batch_begin);
        }
        desc.kernels[reader_kernel_index].emplace_runtime_args(core, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = ring_size * operation_attributes.num_links;

        uint32_t output_tile_id_start =
            row_major ? input_tile_id_start : (input_tile_id_start * num_output_pages) / num_input_pages;
        uint32_t output_tile_id_end =
            row_major ? input_tile_id_end : (input_tile_id_end * num_output_pages) / num_input_pages;
        // page id in gathered tensor with the write page offset
        output_tile_id_start += write_page_offset;
        output_tile_id_end += write_page_offset;

        // The fabric helper appends to a plain std::vector<uint32_t>; we mirror
        // that here, then convert to the variant arg list at emplace time and
        // splice the output_tensor Buffer* binding back at index 0.  The
        // GlobalSemaphore addresses are stable for the workload lifetime
        // (semaphores live on workload_descriptor.semaphores), so they may
        // remain embedded plain uint32 values.
        std::vector<uint32_t> writer_rt_args;
        if (row_major) {
            writer_rt_args = {
                output_tensor.buffer()->address(),
                semaphore.address(),
                output_tile_id_start,
                output_tile_id_end,
                local_height,
                gathered_height,
                ring_index,
                wait_output_semaphore,
                reset_global_semaphore,
                drain_sync_core.x,
                drain_sync_core.y,
                out_ready_sem_wait_value,
                barrier_semaphore.address(),
                barrier_core.x,
                barrier_core.y,
            };
        } else {
            writer_rt_args = {
                output_tensor.buffer()->address(),
                semaphore.address(),
                barrier_semaphore.address(),
                output_tile_id_start,
                output_tile_id_end,
                wait_output_semaphore,
                reset_global_semaphore,
                drain_sync_core.x,
                drain_sync_core.y,
                out_ready_sem_wait_value,
                barrier_core.x,
                barrier_core.y,
            };
        }
        auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
        writer_rt_args.push_back(num_connections);

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

        // Templated fabric helper supports both Program and ProgramDescriptor;
        // here it appends fabric routing args and adds kernel defines to the
        // writer descriptor at writer_kernel_index.
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args<tt::tt_metal::ProgramDescriptor>(
            sender_fabric_node_id, dst_nodes, {link}, desc, writer_kernel_index, core, writer_rt_args);

        // Convert to the variant arg list and substitute the output Buffer*
        // at the first position so the fast cache-hit path patches it.
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> writer_rt_args_variant;
        writer_rt_args_variant.reserve(writer_rt_args.size());
        writer_rt_args_variant.emplace_back(output_tensor.buffer());
        for (size_t i = 1; i < writer_rt_args.size(); ++i) {
            writer_rt_args_variant.emplace_back(writer_rt_args[i]);
        }
        desc.kernels[writer_kernel_index].emplace_runtime_args(core, writer_rt_args_variant);
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AllGatherViaBroadcastFactory::create_workload_descriptor(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0));
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttsl::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Workload-scoped semaphores: allocated once on cache miss and parked on
    // workload_descriptor.semaphores so they outlive the cached workload (the
    // program cache keeps the WorkloadDescriptor alive).  Both semaphores must
    // be ready before any device starts executing, hence the synchronize.
    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    workload_descriptor.semaphores.push_back(init_barrier_semaphore);
    workload_descriptor.semaphores.push_back(final_barrier_semaphore);

    workload_descriptor.programs.reserve(tensor_coords.coords().size());
    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_descriptor_at(
            operation_attributes,
            coord,
            tensor_args.input_tensor,
            output_tensor,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace experimental::prim
}  // namespace ttnn

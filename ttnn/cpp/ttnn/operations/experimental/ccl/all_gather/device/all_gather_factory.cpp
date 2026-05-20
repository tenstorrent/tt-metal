// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>
#include <numeric>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::prim {

using namespace ::ttnn::ccl;

namespace {

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

}  // namespace

AllGatherFactory::cached_mesh_workload_t AllGatherFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args.input_tensor,
            output_tensor,
            final_barrier_semaphore,
            init_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherFactory::cached_program_t AllGatherFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const Tensor& input,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& semaphore,
    const tt::tt_metal::GlobalSemaphore& barrier_semaphore) {
    const auto& input_tensor = input;
    tt::tt_metal::Program program{};

    uint32_t num_devices = operation_attributes.ring_size;
    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    [[maybe_unused]] bool is_first_chip = device_idx == 0;
    [[maybe_unused]] bool is_last_chip = device_idx == num_devices - 1;
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
        num_devices, device_idx, operation_attributes.topology, true);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;

    auto sender_worker_core_range =
        get_cores_close_to_erisc(operation_attributes.num_links * num_workers_per_link, true);
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range);

    // TODO tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t MAX_PACKET_SIZE_BYTES = 6144;
    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();
    uint32_t cb_page_size = std::lcm(std::lcm(input_page_size, output_page_size), MAX_PACKET_SIZE_BYTES);

    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // 32^2 elements == 1/2 or 1 packet, a couple more packets per cb_page for less sync
        cb_page_size *= 4;
    }

    // Per-device page counts are defined by the local shard buffer, not by raw logical dims.
    // Converting pages via logical_shape() breaks tiled tensors and can zero out num_input_pages.
    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    TT_FATAL(num_input_pages > 0, "Broadcast all-gather requires at least one input page");
    TT_FATAL(
        input_tensor.buffer()->aligned_page_size() == input_tensor.buffer()->page_size(),
        "AG doesnt support unaligned RM pages");

    // Compute how many contiguous output pages to write to before jumping, and what the jump
    // (stride) should be.
    // Equal to product of dims after gather dim. Ex: for shape (a, b, c, d, e), if 'c' is the
    // gather dim, then: output_pages_per_stride = c * d * e
    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }
    uint32_t inner_pages = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent = input_shape[i];
        if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
            if (i == rank - 2) {
                extent /= tt::constants::TILE_HEIGHT;
            } else if (i == rank - 1) {
                extent /= tt::constants::TILE_WIDTH;
            }
        } else {
            if (i == rank - 1) {
                extent = 1;
            }
        }
        inner_pages *= extent;
    }
    uint32_t output_pages_per_stride = inner_pages;
    uint32_t output_page_stride = (num_devices - 1) * output_pages_per_stride + 1;
    TT_FATAL(output_pages_per_stride > 0, "output_pages_per_stride must be > 0");

    // Special case: ROW_MAJOR gather on dim -1: pages get wider, not more numerous.
    // Each device needs to write its own portion at some offset within the output page.
    // We emulate this behavior by setting input_page_size as the kernel's output page size,
    // and set output_pages_per_stride = num_input_pages so the iterator produces page IDs
    // like (0, 1, 2, ...).
    uint32_t kernel_output_page_size = output_page_size;
    uint32_t output_page_byte_offset = 0;
    bool is_rm_last_dim = input_tensor.layout() == ttnn::ROW_MAJOR_LAYOUT && gather_dim == rank - 1;
    if (is_rm_last_dim) {
        kernel_output_page_size = input_page_size;
        output_pages_per_stride = num_input_pages;
        output_page_byte_offset = device_idx * input_page_size;
    }
    // For sharded RM tensors, input/output shard widths can differ, making page sizes differ.
    // This scaling converts input page counts to output page counts proportionally.
    const uint32_t num_output_pages = (num_input_pages * input_page_size) / kernel_output_page_size;

    // L1 Scratch CB Creation
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(3 * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        cb0_id,                                                 // cb0_id
        input_page_size,                                        // input tensor page size
        kernel_output_page_size,                                // output page size (input_page_size for RM last-dim)
        output_pages_per_stride,                                // consecutive pages before a stride jump
        output_page_stride,                                     // jump amount at stride boundary
        cb_page_size,                                           // cb entry size
        MAX_PACKET_SIZE_BYTES,                                  // packet_size
        forward_coord.has_value() ? num_targets_forward : 0,    // range_hops
        backward_coord.has_value() ? num_targets_backward : 0,  // range_hops alternate (opposite dir)
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer kernel
    std::vector<uint32_t> writer_compile_args = {
        cb0_id,                                                 // cb0_id
        kernel_output_page_size,                                // output page size (input_page_size for RM last-dim)
        output_pages_per_stride,                                // consecutive pages before a stride jump
        output_page_stride,                                     // jump amount at stride boundary
        cb_page_size,                                           // cb entry size
        MAX_PACKET_SIZE_BYTES,                                  // packet_size
        backward_coord.has_value() ? num_targets_backward : 0,  // range_hops
        forward_coord.has_value() ? num_targets_forward : 0,    // range_hops alternate (opposite dir)
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather/device/kernels/reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args));

    // Writer
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather/device/kernels/writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args));

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto* mesh_device = input_tensor.device();
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set runtime args
        uint32_t input_pages_per_link = num_input_pages / operation_attributes.num_links;
        uint32_t remainder = num_input_pages % operation_attributes.num_links;
        uint32_t input_tile_id_start = (link * input_pages_per_link) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);

        // Map input page range to output page range for this worker, assuming the output
        // tensor only contains our slice.
        // Scaling handles sharded RM tensors where input/output page sizes can differ.
        uint32_t local_output_start = (input_tile_id_start * num_output_pages) / num_input_pages;
        uint32_t local_output_end = (input_tile_id_end * num_output_pages) / num_input_pages;
        uint32_t num_worker_output_pages = local_output_end - local_output_start;
        // Derive output page range in the actual output tensor containing all device slices:
        //       output_page_id = (local / G * N + device_idx) * G + local % G
        // For RM last-dim, page IDs are simply (0,1,2,...) — device position is in byte offset only.
        uint32_t output_page_id_start =
            (local_output_start / output_pages_per_stride * num_devices + device_idx) * output_pages_per_stride +
            local_output_start % output_pages_per_stride;
        uint32_t output_page_in_stride_start = local_output_start % output_pages_per_stride;
        if (is_rm_last_dim) {
            output_page_id_start = local_output_start;
        }

        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = num_devices * operation_attributes.num_links;

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),   // input tensor address
            output_tensor.buffer()->address(),  // output tensor address
            input_tile_id_start,                // input_page_id_start
            input_tile_id_end,                  // input_page_id_end
            output_page_id_start,               // output page start
            output_page_in_stride_start,        // initial position within stride
            output_page_byte_offset,            // byte offset within output page (for RM gather_dim=-1)
            num_worker_output_pages,            // number of output pages for this worker
            device_idx,                         // this device's index
            1,                                  // num_connections // TODO hardcoded
        };
        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        if (forward_coord.has_value()) {
            const auto dst_node = mesh_device->get_fabric_node_id(forward_coord.value());
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                {dst_node},
                {link},
                program,
                worker_sender_reader_kernel_id,
                {core},
                reader_rt_args);
        }

        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // output tensor address
            semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
            barrier_semaphore.address(),        // barrier_sem
            output_page_id_start,               // output page start
            output_page_in_stride_start,        // initial position within stride
            output_page_byte_offset,            // byte offset within output page (for RM gather_dim=-1)
            num_worker_output_pages,            // number of output pages for this worker
            device_idx,                         // this device's index
            wait_output_semaphore,              // wait_output_semaphore
            reset_global_semaphore,             // reset_global_semaphore
            drain_sync_core.x,                  // out_ready_sem_noc0_x
            drain_sync_core.y,                  // out_ready_sem_noc0_y
            out_ready_sem_wait_value,           // out_ready_sem_wait_value
            barrier_core.x,                     // barrier_sem_noc0_x
            barrier_core.y,                     // barrier_sem_noc0_y
            1,                                  // num_connections, // TODO hardcoded
        };

        if (backward_coord.has_value()) {
            const auto dst_node = mesh_device->get_fabric_node_id(backward_coord.value());
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                {dst_node},
                {link},
                program,
                worker_sender_writer_kernel_id,
                {core},
                writer_rt_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = sender_worker_cores,
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .semaphore = semaphore,
        .barrier_semaphore = barrier_semaphore,
        .ring_index = device_idx,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        log_trace(tt::LogOp, "DEBUG: semaphore: {}", shared_vars.semaphore.address());
        log_trace(tt::LogOp, "DEBUG: barrier_semaphore: {}", shared_vars.barrier_semaphore.address());
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);

        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader: [0]=input_addr, [1]=output_addr
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = tensor_args.input_tensor.buffer()->address();
            worker_reader_sender_runtime_args[1] = output_tensor.buffer()->address();
            // writer: [0]=output_addr, [1]=semaphore, [2]=barrier_sem
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = output_tensor.buffer()->address();
            worker_writer_sender_runtime_args[1] = shared_vars.semaphore.address();
            worker_writer_sender_runtime_args[2] = shared_vars.barrier_semaphore.address();
        }
    }
}

}  // namespace ttnn::experimental::prim

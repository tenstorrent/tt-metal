// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>

#include <bit>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::prim {

using namespace ::ttnn::ccl;

namespace {  // anonymous namespace for internal helpers

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

    // TODO only use sems if output not provided. need to wire this down to kernel level.
    // If output tensors are not persistent (globally allocated), we need to wait for all devices to be ready
    // before beginning our operation.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // tt::tt_metal::GlobalSemaphore init_barrier_semaphore;
    // tt::tt_metal::GlobalSemaphore final_barrier_semaphore;
    // if (!tensor_args.persistent_output_tensor.has_value()) {
    // Allocate semaphores in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may cause memory fragmentation and lead to memory allocation failures "
            "for subsequent operations. Configure an L1_SMALL region to avoid this.");
    }

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");
    //}

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

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    ////////////////////////////////////////////////////////////////

    uint32_t num_devices = operation_attributes.ring_size;
    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    std::optional<MeshCoordinate> backward_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "No neighboring devices");

    // Get OP Config, topology config
    auto [num_targets_forward, num_targets_backward] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
        num_devices, device_idx, operation_attributes.topology, true);
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;

    auto sender_worker_core_range =
        get_cores_close_to_erisc(operation_attributes.num_links * num_workers_per_link, true);
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range);

    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // Glossary:
    //   input page     -- one page of the input tensor.
    //   output page    -- one page of the output tensor.
    //                     In concat mode the *kernel-visible* output page size is
    //                     smaller than the tensor's actual output page.
    //   stripe         -- a run of consecutive output page ids this device writes
    //                     before jumping past other devices' contributions.
    //   stripe jump    -- value the kernel adds to output_page_id at the stripe
    //                     boundary.
    //   stripe distance-- page-id distance from start of one of this device's
    //                     stripes to the start of the next.
    //
    // Three copy modes, picked by input vs output page sizes:
    //   matched (in == out): 1 write per input page, byte offset = 0.
    //   concat  (out > in) : 1 write per input page, byte offset = (d % concat_factor) * in.
    //   split   (in > out) : split_factor writes per input page, byte offset = 0.
    //
    // Kernel is a dumb page iterator. Iteration pattern is periodic stripes, i.e.
    // consecutive pages (stripe) followed by periodic jumps (to the next stripe).
    //
    // Host derives the iterator parameters from input/output page sizes, gather dim,
    // and device index.
    ////////////////////////////////////////////////////////////////

    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();

    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

    // --- Copy mode ---
    // Exactly one of concat_factor or split_factor is > 1 (or both = 1 in matched mode).
    const uint32_t concat_factor = std::max(1u, output_page_size / input_page_size);
    const uint32_t split_factor = std::max(1u, input_page_size / output_page_size);

    // kernel_output_page_size = bytes per write = min(input, output).
    const uint32_t kernel_output_page_size = std::min(input_page_size, output_page_size);
    const uint32_t output_page_byte_offset = (device_idx % concat_factor) * input_page_size;
    const uint32_t device_slot = device_idx / concat_factor;
    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_pages = num_input_pages * split_factor;

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // kernel_output_page_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;

    // Perf hack: for tile layout, pack multiple pages into a single CB page to reduce
    // CB sync frequency between reader and writer. Don't do this for RM because of all
    // the careful handling of page sizes.
    // TODO identify the multiplier based on available L1 space
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        cb_page_size *= 4;
    }

    // --- Stripe geometry ---
    // input_pages_per_stripe = num input pages along [gather_dim .. rank-1] this
    // device contributes per stripe. For RM gather_dim=-1 this is the *page* count,
    // which handles sharded RM input (> 1 input page per row).
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tt::constants::TILE_WIDTH;
            } else {
                extent = (input_shape[i] * input_tensor.element_size()) / input_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
            extent = input_shape[i] / tt::constants::TILE_HEIGHT;
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }

    // In concat mode each iter goes to a different output page (one chunk per row
    // from this device), so the stripe is 1 long. Otherwise the stripe spans this
    // device's full contribution: input_pages_per_stripe * split_factor consecutive
    // output pages.
    const uint32_t output_pages_per_stripe = (concat_factor > 1) ? 1u : input_pages_per_stripe * split_factor;
    const uint32_t output_page_stripe_distance = (num_devices * input_pages_per_stripe * split_factor) / concat_factor;
    const uint32_t output_page_stripe_jump = output_page_stripe_distance - output_pages_per_stripe + 1;
    TT_FATAL(output_pages_per_stripe > 0, "output_pages_per_stripe must be > 0");

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

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
        kernel_output_page_size,                                // kernel-visible page size = min(input, output)
        output_pages_per_stripe,                                // stripe length (writes before a stripe jump)
        output_page_stripe_jump,                                // value added to page_id at stripe boundary
        cb_page_size,                                           // cb entry size
        packet_size,                                            // packet_size
        forward_coord.has_value() ? num_targets_forward : 0,    // range_hops
        backward_coord.has_value() ? num_targets_backward : 0,  // range_hops alternate (opposite dir)
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer kernel
    std::vector<uint32_t> writer_compile_args = {
        cb0_id,                                                 // cb0_id
        kernel_output_page_size,                                // kernel-visible page size = min(input, output)
        output_pages_per_stripe,                                // stripe length (writes before a stripe jump)
        output_page_stripe_jump,                                // value added to page_id at stripe boundary
        cb_page_size,                                           // cb entry size
        packet_size,                                            // packet_size
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

        // Map this worker's slice of input pages to its slice of output writes.
        // num_output_pages already accounts for split_factor, so in matched/concat
        // modes the ratio cancels back to num_input_pages.
        uint32_t local_output_start = (input_tile_id_start * num_output_pages) / num_input_pages;
        uint32_t local_output_end = (input_tile_id_end * num_output_pages) / num_input_pages;
        uint32_t num_worker_output_pages = local_output_end - local_output_start;
        // Map local write index -> global output page id of this device's first write:
        //     stripe_index           = local / output_pages_per_stripe
        //     position_in_stripe     = local % output_pages_per_stripe
        //     output_page_id_start   = stripe_index * output_page_stripe_distance
        //                            + device_slot * output_pages_per_stripe
        //                            + position_in_stripe
        // Concat with concat_factor=N collapses naturally (stripe_distance=1, device_slot=0).
        uint32_t output_page_id_start = (local_output_start / output_pages_per_stripe) * output_page_stripe_distance +
                                        device_slot * output_pages_per_stripe +
                                        local_output_start % output_pages_per_stripe;
        uint32_t output_page_in_stripe_start = local_output_start % output_pages_per_stripe;

        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = num_devices * operation_attributes.num_links;

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),   // input tensor address
            output_tensor.buffer()->address(),  // output tensor address
            input_tile_id_start,                // input_page_id_start
            input_tile_id_end,                  // input_page_id_end
            output_page_id_start,               // output page start
            output_page_in_stripe_start,        // initial position within stripe
            output_page_byte_offset,            // byte offset within output page (nonzero only in page-concat)
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
            output_page_in_stripe_start,        // initial position within stripe
            output_page_byte_offset,            // byte offset within output page (nonzero only in page-concat)
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

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::experimental::prim {

using namespace ::ttnn::ccl;

// TODO finalize core placement.
/*
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
*/

AllGatherFactory::cached_mesh_workload_t AllGatherFactory::create_mesh_workload(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice_id);
    if (operation_attributes.sub_core_grid.has_value()) {
        available_cores = available_cores.intersection(operation_attributes.sub_core_grid.value());
    }
    ttnn::SmallVector<tt::tt_metal::SubDeviceId> subdevices = {subdevice_id};

    // Kernel needs to wait to receive all remote data before exiting, and in some cases needs to wait
    // for all remote devices to be ready before beginning operation.
    // Since Fabric doesn't provide such capability within kernels, we need to manually sync using global semaphores.
    // Allocate the semaphore in L1_SMALL to avoid fragmenting the larger L1 memory pool.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_warning(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and cause allocation failures for subsequent "
            "operations. Configure an L1_SMALL region to avoid this.");
    }
    auto barrier_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphore allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, output_tensor, barrier_sem);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

AllGatherFactory::cached_program_t AllGatherFactory::create_at(
    const AllGatherParams& operation_attributes,
    const ttnn::MeshCoordinate& sender_device_coord,
    const AllGatherInputs& tensor_args,
    const Tensor& output_tensor,
    const tt::tt_metal::GlobalSemaphore& barrier_sem) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};

    ////////////////////////////////////////////////////////////////
    // Fabric setup
    ////////////////////////////////////////////////////////////////

    const uint32_t num_devices = operation_attributes.num_devices;
    uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);
    // TODO verify row-major device_idx matches ShardTensorToMesh order under 2D no-cluster_axis;
    // manual (2,4) test will catch any mismatch. i.e. in 2D mesh shape, whats the device_slot?

    // Compute hops + neighbors for each mesh axis.
    // Each axis ∈ {0, 1} contributes a forward/backward pair: axis 1 -> (E=fwd, W=bwd),
    // axis 0 -> (S=fwd, N=bwd). In 1D only one axis is active; in 2D both can be.
    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());

    std::optional<MeshCoordinate> e_coord, w_coord, n_coord, s_coord;
    uint32_t e_hops = 0, w_hops = 0, n_hops = 0, s_hops = 0;
    bool ew_load_balance = false;
    bool ns_load_balance = false;

    for (uint32_t axis = 0; axis < 2; ++axis) {
        const uint32_t axis_size = operation_attributes.axis_num_devices[axis];
        const bool is_axis_active = axis_size > 1;
        if (!is_axis_active) {
            continue;
        }

        const auto axis_topology = operation_attributes.axis_topology[axis];
        const uint32_t axis_index = sender_device_coord[axis];
        auto [fwd_hops, bwd_hops] = ::ttnn::ccl::get_forward_backward_line_mcast_distance(
            axis_size, axis_index, axis_topology, /*static_alternate=*/false);
        auto fwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, 1, axis_topology, axis);
        auto bwd_coord = ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, sender_device_coord, -1, axis_topology, axis);

        // A load-balancing technique (alternating between two imbalanced routes) is used
        // in even-sized rings.
        const bool axis_load_balance = tt::tt_fabric::is_ring_or_torus(axis_topology) && (axis_size % 2 == 0);
        if (axis == 1) {
            e_hops = fwd_hops;
            w_hops = bwd_hops;
            e_coord = fwd_coord;
            w_coord = bwd_coord;
            ew_load_balance = axis_load_balance;
        } else {
            s_hops = fwd_hops;
            n_hops = bwd_hops;
            s_coord = fwd_coord;
            n_coord = bwd_coord;
            ns_load_balance = axis_load_balance;
        }
    }
    TT_FATAL(
        e_coord.has_value() || w_coord.has_value() || n_coord.has_value() || s_coord.has_value(),
        "No neighboring devices");

    // We allocate one worker core per link, but not per axis.
    // Each worker handles both dirs (forward and backward) and also both axes (N/S and E/W).
    // Known limitation: when the two axes have unequal link counts, the larger axis's extra links
    // go unused. If this is ever a real use-case, we need to allocate separate worker cores per axis.
    const uint32_t links0 = operation_attributes.axis_num_links[0];
    const uint32_t links1 = operation_attributes.axis_num_links[1];
    const uint32_t min_num_links = std::min(links0 > 0 ? links0 : links1, links1 > 0 ? links1 : links0);

    // Get worker cores
    uint32_t num_workers_per_link = 1;
    // TODO finalize core placement
    /*auto sender_worker_core_range = get_cores_close_to_erisc(min_num_links * num_workers_per_link, true);
    auto sender_worker_cores = corerange_to_cores(sender_worker_core_range);*/
    auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        min_num_links,
        num_workers_per_link,
        input_tensor.device(),
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid);

    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    // Kernel alternates between ranges[] and ranges_alt[] hops on every packet send.
    // Enabled if any axis is an even-sized ring.
    const bool load_balance_across_alt_routes = ew_load_balance || ns_load_balance;

    // We use an init barrier to wait for remote output tensors to be allocated. This only
    // matters when the output is freshly allocated by the op; a persistent/preallocated
    // output is guaranteed to already exist on every device before op kernel begins.
    const bool do_init_barrier = !tensor_args.persistent_output_tensor.has_value();

    ////////////////////////////////////////////////////////////////
    // Page indexing
    //
    // Glossary:
    //   input page     -- one page of the input tensor.
    //   output page    -- one page of the output tensor (the real buffer page).
    //   chunk          -- one NOC write = min(input_page, output_page) bytes. An input
    //                     page = split_factor chunks; an output page = output_chunks_per_page
    //                     chunks. The kernel iterator walks chunks.
    //   stripe         -- a run of consecutive chunks this device writes before
    //                     jumping past other devices' contributions.
    //   stripe jump    -- value the kernel adds to output_page_id at the stripe
    //                     boundary.
    //
    // Three copy modes, picked by input vs output page sizes:
    //   matched (in == out): 1 chunk per input page, output_chunks_per_page = 1.
    //   concat  (out > in) : 1 chunk per input page, output_chunks_per_page > 1; each
    //                        chunk lands at a byte offset within a shared output page.
    //   split   (in > out) : split_factor chunks per input page, output_chunks_per_page = 1.
    //
    // Kernel is a dumb chunk iterator. Iteration pattern is:
    //   byte_offset++ within an output page -> chunk++ -> stripe+=jump
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
    // output_chunks_per_page > 1 in concat mode; split_factor > 1 in split mode; both
    // = 1 in matched mode (output_chunk_size == input_page_size == output_page_size).
    const uint32_t output_chunk_size = std::min(input_page_size, output_page_size);  // NOC write size
    const uint32_t output_chunks_per_page = std::max(1u, output_page_size / input_page_size);
    const uint32_t split_factor = std::max(1u, input_page_size / output_page_size);

    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_chunks = num_input_pages * split_factor;

    // --- CB sizing ---
    // cb_page_size is a multiple of input_page_size, which is itself a multiple of
    // output_chunk_size = min(input, output), so the kernel increments both
    // the cb_read_ptr and cb_write_ptr cleanly.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;
    uint32_t cb_depth = 3;  // NOTE: reader's txn ID push/pop scheme has only been tested with depth=3

    // Perf hack: for tile layout, pack multiple pages into a single CB page to reduce CB sync
    // frequency between reader and writer. Note this increases effective CB depth.
    // Don't do this for row-major layout because of all the careful handling of page sizes.
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // Empirically determined heuristic, works well for all tensor sizes
        const uint32_t ideal_multiplier = (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) ? 4 : 3;
        // Find the largest multiplier in [1, ideal] that fits in available L1
        const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, ideal_multiplier);
        if (multiplier < ideal_multiplier) {
            log_warning(
                tt::LogOp,
                "CircularBuffer depth is reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
        }
        cb_page_size *= multiplier;
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

    // Stripe = this device's contiguous run of chunks per row = input_pages_per_stripe
    // * split_factor. Measured in chunks (not output pages) so multi-shard concat works:
    // a stripe's chunks are laid across output pages via the inner byte-offset counter
    // and may straddle pages.
    const uint32_t output_chunks_per_stripe = input_pages_per_stripe * split_factor;
    const uint32_t stripe_distance_chunks = num_devices * output_chunks_per_stripe;
    const uint32_t output_pages_per_row = stripe_distance_chunks / output_chunks_per_page;
    // This device's chunk phase within the output page. Constant across rows because
    // output_chunks_per_page divides stripe_distance_chunks (valid output sharding).
    const uint32_t off_start_chunks = (device_idx * output_chunks_per_stripe) % output_chunks_per_page;
    // Page carries accumulated while walking one full stripe.
    const uint32_t in_stripe_carries = (off_start_chunks + output_chunks_per_stripe - 1) / output_chunks_per_page;
    // Value added to output_page_id at the stripe boundary (jump to this device's run
    // in the next row): pages_per_row minus the carries already taken within the stripe.
    const uint32_t output_page_stripe_jump = output_pages_per_row - in_stripe_carries;
    // Per-device byte offset phase the iterator resets to at each stripe boundary.
    const uint32_t output_page_byte_offset = off_start_chunks * output_chunk_size;
    TT_FATAL(output_chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");

    ////////////////////////////////////////////////////////////////
    // Circular Buffer and Kernel creation
    ////////////////////////////////////////////////////////////////

    // L1 Scratch CB Creation
    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // KERNEL CREATION
    // Reader (covers forward directions E-line + S-rect)
    std::vector<uint32_t> reader_compile_args = {
        cb0_id,                          // cb0_id
        input_page_size,                 // input tensor page size
        output_chunk_size,               // NOC write size = min(input, output)
        output_chunks_per_page,          // chunks per output buffer page (1 unless concat)
        output_chunks_per_stripe,        // stripe length in chunks (before a stripe jump)
        output_page_stripe_jump,         // value added to output_page_id at stripe boundary
        cb_page_size,                    // cb entry size
        packet_size,                     // packet_size
        load_balance_across_alt_routes,  // load_balance_across_alt_routes
        (e_hops > 0) + (s_hops > 0),     // num_connections
        do_init_barrier,                 // do_init_barrier
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_args);

    // Writer (covers backward directions W-line + N-rect)
    std::vector<uint32_t> writer_compile_args = {
        cb0_id,                          // cb0_id
        output_chunk_size,               // NOC write size = min(input, output)
        output_chunks_per_page,          // chunks per output buffer page (1 unless concat)
        output_chunks_per_stripe,        // stripe length in chunks (before a stripe jump)
        output_page_stripe_jump,         // value added to output_page_id at stripe boundary
        cb_page_size,                    // cb entry size
        packet_size,                     // packet_size
        load_balance_across_alt_routes,  // load_balance_across_alt_routes
        (w_hops > 0) + (n_hops > 0),     // num_connections
        do_init_barrier,                 // do_init_barrier
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
    auto* mesh_device = input_tensor.device();
    for (uint32_t link = 0; link < min_num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

        // Set runtime args
        uint32_t input_pages_per_link = num_input_pages / min_num_links;
        uint32_t remainder = num_input_pages % min_num_links;
        uint32_t input_tile_id_start = (link * input_pages_per_link) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);

        // Map this worker's slice of input pages to its slice of output chunks.
        // num_output_chunks already accounts for split_factor, so in matched/concat
        // modes the ratio cancels back to num_input_pages.
        uint32_t local_output_start = (input_tile_id_start * num_output_chunks) / num_input_pages;
        uint32_t local_output_end = (input_tile_id_end * num_output_chunks) / num_input_pages;
        uint32_t num_worker_output_chunks = local_output_end - local_output_start;
        // s_start = global chunk index of this worker's first write:
        //     stripe_index  = local / output_chunks_per_stripe
        //     pos_in_stripe = local % output_chunks_per_stripe
        //     s_start       = stripe_index * stripe_distance_chunks    (skip other devices' rows)
        //                   + device_idx   * output_chunks_per_stripe  (this device's run in the row)
        //                   + pos_in_stripe
        uint32_t s_start = (local_output_start / output_chunks_per_stripe) * stripe_distance_chunks +
                           device_idx * output_chunks_per_stripe + local_output_start % output_chunks_per_stripe;
        uint32_t output_page_id_start = s_start / output_chunks_per_page;
        uint32_t output_page_byte_offset_start = (s_start % output_chunks_per_page) * output_chunk_size;
        uint32_t output_chunk_in_stripe_start = local_output_start % output_chunks_per_stripe;

        // Per-link barrier fan-in = N-1 in every case. Every other chip sends me one atomic_inc:
        //   1D: e_hops + w_hops (or n_hops + s_hops) along the active axis = axis_size - 1.
        //   2D: every chip in the mesh outside me is covered by exactly one of the 4 mcast packets.
        // Both equal num_devices - 1.
        uint32_t barrier_wait_value = num_devices - 1;

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),   // input tensor address
            output_tensor.buffer()->address(),  // output tensor address
            input_tile_id_start,                // input_page_id_start
            input_tile_id_end,                  // input_page_id_end
            output_page_id_start,               // output page start
            output_chunk_in_stripe_start,       // initial chunk position within stripe
            output_page_byte_offset,            // per-device offset phase (reset at stripe boundary)
            output_page_byte_offset_start,      // worker's initial byte offset within output page
            num_worker_output_chunks,           // number of output chunks for this worker
            device_idx,                         // this device's index
            barrier_sem.address(),              // barrier_sem L1 address
            virtual_core.x,                     // barrier_sem location (core.x)
            virtual_core.y,                     // barrier_sem location (core.y)
            barrier_wait_value,                 // barrier counter to wait for
            e_hops,                             // line_hops
            e_hops,                             // rect_e_hops
            w_hops,                             // rect_w_hops
            s_hops,                             // rect_spine_hops
            ew_load_balance ? w_hops : e_hops,  // line_hops_alt
            ew_load_balance ? w_hops : e_hops,  // rect_e_hops_alt
            ew_load_balance ? e_hops : w_hops,  // rect_w_hops_alt
            ns_load_balance ? n_hops : s_hops,  // rect_spine_hops_alt
        };
        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        // Reader forward connection info: E-line (axis 1) then S-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> reader_dst_nodes;
        if (e_hops > 0 && e_coord.has_value()) {
            reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*e_coord));
        }
        if (s_hops > 0 && s_coord.has_value()) {
            reader_dst_nodes.push_back(mesh_device->get_fabric_node_id(*s_coord));
        }
        if (!reader_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                reader_dst_nodes,
                {link},
                program,
                worker_sender_reader_kernel_id,
                {core},
                reader_rt_args,
                fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        }

        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // output tensor address
            output_page_id_start,               // output page start
            output_chunk_in_stripe_start,       // initial chunk position within stripe
            output_page_byte_offset,            // per-device offset phase (reset at stripe boundary)
            output_page_byte_offset_start,      // worker's initial byte offset within output page
            num_worker_output_chunks,           // number of output chunks for this worker
            device_idx,                         // this device's index
            barrier_sem.address(),              // barrier_sem L1 address
            virtual_core.x,                     // barrier_sem location (core.x)
            virtual_core.y,                     // barrier_sem location (core.y)
            w_hops,                             // line_hops
            e_hops,                             // rect_e_hops
            w_hops,                             // rect_w_hops
            n_hops,                             // rect_spine_hops
            ew_load_balance ? e_hops : w_hops,  // line_hops_alt
            ew_load_balance ? w_hops : e_hops,  // rect_e_hops_alt
            ew_load_balance ? e_hops : w_hops,  // rect_w_hops_alt
            ns_load_balance ? s_hops : n_hops,  // rect_spine_hops_alt
        };

        // Writer backward connections: W-line (axis 1) then N-rect (axis 0).
        std::vector<tt::tt_fabric::FabricNodeId> writer_dst_nodes;
        if (w_hops > 0 && w_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*w_coord));
        }
        if (n_hops > 0 && n_coord.has_value()) {
            writer_dst_nodes.push_back(mesh_device->get_fabric_node_id(*n_coord));
        }
        if (!writer_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                sender_fabric_node_id,
                writer_dst_nodes,
                {link},
                program,
                worker_sender_writer_kernel_id,
                {core},
                writer_rt_args,
                fabric_is_2d ? tt::tt_fabric::FabricApiType::Mesh : tt::tt_fabric::FabricApiType::Linear);
        }

        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    shared_variables_t shared_variables{
        .sender_worker_cores = sender_worker_cores,
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .barrier_sem = barrier_sem,
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

        const uint32_t barrier_sem_addr = shared_vars.barrier_sem.address();
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);

        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader: [0]=input_addr, [1]=output_addr, [10]=barrier_sem
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = tensor_args.input_tensor.buffer()->address();
            worker_reader_sender_runtime_args[1] = output_tensor.buffer()->address();
            worker_reader_sender_runtime_args[10] = barrier_sem_addr;
            // writer: [0]=output_addr, [7]=barrier_sem
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = output_tensor.buffer()->address();
            worker_writer_sender_runtime_args[7] = barrier_sem_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim

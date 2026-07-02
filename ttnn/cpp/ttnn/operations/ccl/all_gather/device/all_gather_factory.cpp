// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_factory.hpp"

#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

namespace ttnn::operations::ccl {

using namespace ::ttnn::ccl;

////////////////////////////////////////////////////////////////////////////////
// Store-and-forward AllGather (Fabric_1D line/ring only).
//
// Every device relays stripes to its immediate neighbor one hop at a time; a shard reaches far devices by
// being re-forwarded at each hop. Forward and backward conveyors run on separate cores. Each conveyor: the
// reader (CB producer, no fabric) reads iteration 0 from this device's own input and later iterations from
// the stripe the upstream neighbor deposited into our output; the writer (CB consumer) unicasts each stripe
// one hop to the neighbor's output (same address on every device).
//
// Direction/topology/load-balance are decided here and passed as runtime args, so the two kernels are
// compiled once and run on both core sets. Two semaphores: data_valid (per-iteration relay gate +
// completion, mirror-core targeted) and ready (init handshake, opposite-core targeted). Readers own every
// wait; writers only send.
////////////////////////////////////////////////////////////////////////////////

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

    // Neighbor-to-neighbor sync uses two per-core semaphores: data_valid (a stripe has been relayed in) and
    // ready (the init handshake). Allocate in L1_SMALL to avoid fragmenting the larger L1 pool.
    bool l1_small_size = mesh_device->allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    auto sem_buffer_type = l1_small_size > 0 ? tt::tt_metal::BufferType::L1_SMALL : tt::tt_metal::BufferType::L1;
    if (sem_buffer_type != tt::tt_metal::BufferType::L1_SMALL) {
        log_info(
            tt::LogOp,
            "Allocating semaphores in L1, which may fragment L1 and reduce headroom for subsequent op "
            "allocations. Configure an L1_SMALL region to mitigate this.");
    }
    auto data_valid_sem =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    auto ready_sem = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0, sem_buffer_type);
    log_debug(tt::LogOp, "Semaphores allocated and waiting for all devices to be ready");
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);
    log_debug(tt::LogOp, "All devices are ready, starting program execution");

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_args, output_tensor, data_valid_sem, ready_sem);
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
    const tt::tt_metal::GlobalSemaphore& data_valid_sem,
    const tt::tt_metal::GlobalSemaphore& ready_sem) {
    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::Program program{};
    auto* mesh_device = input_tensor.device();

    ////////////////////////////////////////////////////////////////
    // Topology (Fabric_1D line/ring only)
    ////////////////////////////////////////////////////////////////

    const bool fabric_is_2d = ::tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig());
    TT_FATAL(!fabric_is_2d, "store-and-forward all_gather supports Fabric_1D line/ring only, not Fabric_2D");

    // Exactly one active axis in 1D (the other has num_devices == 1).
    int active_axis = -1;
    for (uint32_t a = 0; a < 2; ++a) {
        if (operation_attributes.axis_num_devices[a] > 1) {
            TT_FATAL(active_axis == -1, "store-and-forward all_gather supports exactly one active axis (1D)");
            active_axis = static_cast<int>(a);
        }
    }
    TT_FATAL(active_axis != -1, "No neighboring devices");
    const uint32_t axis = static_cast<uint32_t>(active_axis);
    const auto topology = operation_attributes.axis_topology[axis];
    const bool is_ring = tt::tt_fabric::is_ring_or_torus(topology);

    const uint32_t num_devices = operation_attributes.num_devices;
    const uint32_t device_idx = ::ttnn::ccl::get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);

    auto fwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, 1, topology, axis);
    auto bwd_coord =
        ::ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, sender_device_coord, -1, topology, axis);

    // Stripes each conveyor sends: ring -> N/2; line fwd -> d+1, bwd -> N-d; 0 at a dead endpoint. The rest of
    // each conveyor's schedule (num_recv, stripe_step, do_local_write, half-slice) is derived per conveyor below.
    const uint32_t fwd_iters = is_ring ? num_devices / 2 : (device_idx + 1 < num_devices ? device_idx + 1 : 0);
    const uint32_t bwd_iters = is_ring ? num_devices / 2 : (device_idx > 0 ? num_devices - device_idx : 0);
    TT_FATAL(fwd_iters > 0 || bwd_iters > 0, "device participates in neither direction");

    // Even ring: the last iteration sends a contiguous half of the stripe so the two conveyors split the
    // antipode slice (forward first half, backward second half). N==2 rings are remapped to line upstream, so
    // an even ring here has num_iters>=2 and the split never lands on iteration 0.
    const bool ring_even_split = is_ring && (num_devices % 2 == 0) && (num_devices >= 4);
    const bool do_init_barrier = !tensor_args.persistent_output_tensor.has_value();

    const uint32_t num_links = operation_attributes.axis_num_links[axis];

    ////////////////////////////////////////////////////////////////
    // Worker cores: one forward + one backward core per link. Allocate all at once; the first num_links are the
    // forward conveyors, the next num_links the backward ones.
    ////////////////////////////////////////////////////////////////

    auto [worker_core_range, worker_cores] = ttnn::ccl::choose_worker_cores(
        2 * num_links,
        /*num_workers_per_link=*/1,
        mesh_device,
        operation_attributes.subdevice_id,
        /*core_grid_offset=*/CoreCoord{0, 0},
        operation_attributes.sub_core_grid);

    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();

    ////////////////////////////////////////////////////////////////
    // Page indexing (unchanged from the multicast op). See the OutputStripeIterator in common.hpp: the kernel
    // derives per-stripe page/byte/jump from {stripe, slice} and the CT geometry, so the host only supplies
    // the geometry constants and each worker's [local_output_start, num_worker_output_chunks) slice.
    //
    //   input page  -- one page of the input tensor.       output page -- one page of the output tensor.
    //   chunk       -- one NOC write = min(input_page, output_page) bytes.
    //   stripe      -- a device's contiguous run of chunks per row.
    //   Copy modes: matched (in==out), concat (out>in, chunks share a page), split (in>out).
    ////////////////////////////////////////////////////////////////

    const uint32_t input_page_size = input_tensor.buffer()->aligned_page_size();
    const uint32_t output_page_size = output_tensor.buffer()->aligned_page_size();

    auto input_shape = input_tensor.padded_shape();
    uint32_t rank = input_shape.rank();
    int32_t gather_dim = operation_attributes.dim;
    if (gather_dim < 0) {
        gather_dim += rank;
    }

    const uint32_t output_chunk_size = std::min(input_page_size, output_page_size);  // NOC write size
    const uint32_t output_chunks_per_page = std::max(1u, output_page_size / input_page_size);
    const uint32_t split_factor = std::max(1u, input_page_size / output_page_size);

    const uint32_t num_input_pages = input_tensor.buffer()->num_pages();
    const uint32_t num_output_chunks = num_input_pages * split_factor;

    // CB sizing: cb_page_size is a multiple of both input_page_size and output_chunk_size.
    const uint32_t pages_per_packet = std::max(1u, packet_size / input_page_size);
    uint32_t cb_page_size = input_page_size * pages_per_packet;
    uint32_t cb_depth = 3;
    if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
        // Pack multiple pages per CB page (fewer reader/writer syncs). Empirical multiplier, L1-clamped.
        const uint32_t ideal_multiplier = (input_tensor.device()->arch() == tt::ARCH::BLACKHOLE) ? 4 : 3;
        const uint32_t max_l1_space = ttnn::operations::data_movement::get_max_l1_space(input_tensor);
        const uint32_t multiplier = std::clamp(max_l1_space / (cb_depth * cb_page_size), 1u, ideal_multiplier);
        if (multiplier < ideal_multiplier) {
            log_info(
                tt::LogOp,
                "CircularBuffer depth is reduced due to L1 pressure (only {} B available), performance may regress.",
                max_l1_space);
        }
        cb_page_size *= multiplier;
    }

    // Stripe = this device's contiguous run of chunks per row = input pages along [gather_dim..rank-1] * split.
    auto tile = (input_tensor.layout() == Layout::TILE) ? input_tensor.tensor_spec().tile() : tt::tt_metal::Tile();
    uint32_t input_pages_per_stripe = 1;
    for (int32_t i = gather_dim; i < rank; i++) {
        uint32_t extent;
        if (i == rank - 1) {
            if (input_tensor.layout() == ttnn::TILE_LAYOUT) {
                extent = input_shape[i] / tile.get_width();
            } else {
                extent = (input_shape[i] * input_tensor.element_size()) / input_page_size;
            }
        } else if (input_tensor.layout() == ttnn::TILE_LAYOUT && i == rank - 2) {
            extent = input_shape[i] / tile.get_height();
        } else {
            extent = input_shape[i];
        }
        input_pages_per_stripe *= extent;
    }
    const uint32_t output_chunks_per_stripe = input_pages_per_stripe * split_factor;
    TT_FATAL(output_chunks_per_stripe > 0, "output_chunks_per_stripe must be > 0");

    ////////////////////////////////////////////////////////////////
    // Circular buffer + kernels (one reader + one writer, shared across both directions).
    ////////////////////////////////////////////////////////////////

    uint32_t cb0_id = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_depth * cb_page_size, {{cb0_id, df}}).set_page_size(cb0_id, cb_page_size);
    CreateCircularBuffer(program, worker_core_range, cb_src0_config);

    std::vector<uint32_t> reader_ct = {
        input_page_size,
        output_chunk_size,
        output_chunks_per_page,
        output_chunks_per_stripe,
        num_devices,
        cb0_id,
        cb_page_size,
        do_init_barrier ? 1u : 0u,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_ct);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_ct);

    std::vector<uint32_t> writer_ct = {
        output_chunk_size,
        output_chunks_per_page,
        output_chunks_per_stripe,
        num_devices,
        cb0_id,
        cb_page_size,
        packet_size,
        do_init_barrier ? 1u : 0u,
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/reader.cpp",
        worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct));
    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/writer.cpp",
        worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_ct));

    ////////////////////////////////////////////////////////////////
    // Runtime args, per link. The per-link page split is direction-independent.
    ////////////////////////////////////////////////////////////////

    const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (uint32_t link = 0; link < num_links; link++) {
        const uint32_t input_pages_per_link = num_input_pages / num_links;
        const uint32_t remainder = num_input_pages % num_links;
        const uint32_t input_tile_id_start = (link * input_pages_per_link) + std::min(link, remainder);
        const uint32_t input_tile_id_end = ((link + 1) * input_pages_per_link) + std::min(link + 1, remainder);
        const uint32_t local_output_start = (input_tile_id_start * num_output_chunks) / num_input_pages;
        const uint32_t local_output_end = (input_tile_id_end * num_output_chunks) / num_input_pages;
        const uint32_t num_worker_output_chunks = local_output_end - local_output_start;
        const uint32_t half = num_worker_output_chunks / 2;

        // Configure one conveyor. own_vc is this core's coords, reused as the data_valid target on the
        // neighbor's mirror core; partner_vc is the opposite-direction core, the ready target.
        auto set_conveyor = [&](bool is_forward) {
            const CoreCoord core = worker_cores[is_forward ? link : num_links + link];
            const CoreCoord partner = worker_cores[is_forward ? num_links + link : link];
            const CoreCoord own_vc = mesh_device->worker_core_from_logical_core(core);
            const CoreCoord partner_vc = mesh_device->worker_core_from_logical_core(partner);
            const auto& neighbor = is_forward ? fwd_coord : bwd_coord;

            const uint32_t stripe_step = is_forward ? num_devices - 1 : 1;
            const uint32_t num_iters = is_forward ? fwd_iters : bwd_iters;
            const uint32_t num_recv =
                is_ring ? num_devices / 2 : (is_forward ? device_idx : num_devices - 1 - device_idx);
            const bool do_local_write = is_forward ? (fwd_iters > 0) : (fwd_iters == 0);

            uint32_t final_start = local_output_start;
            uint32_t final_count = num_worker_output_chunks;
            if (ring_even_split) {
                final_start = is_forward ? local_output_start : (local_output_start + half);
                final_count = is_forward ? half : (num_worker_output_chunks - half);
            }

            std::vector<uint32_t> reader_rt = {
                input_addr,
                output_addr,
                device_idx,
                stripe_step,
                num_iters,
                num_recv,
                local_output_start,
                num_worker_output_chunks,
                final_start,
                final_count,
                input_tile_id_start,
                input_tile_id_end,
                data_valid_sem.address(),
                ready_sem.address(),
            };
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, {core}, reader_rt);

            std::vector<uint32_t> writer_rt = {
                output_addr,
                device_idx,
                stripe_step,
                num_iters,
                local_output_start,
                num_worker_output_chunks,
                final_start,
                final_count,
                do_local_write ? 1u : 0u,
                data_valid_sem.address(),
                ready_sem.address(),
                (uint32_t)own_vc.x,
                (uint32_t)own_vc.y,
                (uint32_t)partner_vc.x,
                (uint32_t)partner_vc.y,
            };
            if (num_iters > 0 && neighbor.has_value()) {
                std::vector<tt::tt_fabric::FabricNodeId> dst = {mesh_device->get_fabric_node_id(*neighbor)};
                append_routing_plane_connection_manager_rt_args(
                    sender_fabric_node_id,
                    dst,
                    {link},
                    program,
                    writer_kernel_id,
                    {core},
                    writer_rt,
                    tt::tt_fabric::FabricApiType::Linear);
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, {core}, writer_rt);
        };

        set_conveyor(/*is_forward=*/true);
        set_conveyor(/*is_forward=*/false);
    }

    shared_variables_t shared_variables{
        .worker_cores = worker_cores,
        .reader_kernel_id = reader_kernel_id,
        .writer_kernel_id = writer_kernel_id,
        .data_valid_sem = data_valid_sem,
        .ready_sem = ready_sem,
    };

    return {std::move(program), std::move(shared_variables)};
}

void AllGatherFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherParams& /*operation_attributes*/,
    const AllGatherInputs& tensor_args,
    Tensor& output_tensor) {
    const uint32_t input_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);
        const uint32_t data_valid_addr = shared_vars.data_valid_sem.address();
        const uint32_t ready_addr = shared_vars.ready_sem.address();

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        for (const auto& core : shared_vars.worker_cores) {
            // reader: [0]=input_addr, [1]=output_addr, [12]=data_valid, [13]=ready
            auto& r = reader_args_by_core[core.x][core.y];
            r[0] = input_addr;
            r[1] = output_addr;
            r[12] = data_valid_addr;
            r[13] = ready_addr;
            // writer: [0]=output_addr, [9]=data_valid, [10]=ready
            auto& w = writer_args_by_core[core.x][core.y];
            w[0] = output_addr;
            w[9] = data_valid_addr;
            w[10] = ready_addr;
        }
    }
}

}  // namespace ttnn::operations::ccl

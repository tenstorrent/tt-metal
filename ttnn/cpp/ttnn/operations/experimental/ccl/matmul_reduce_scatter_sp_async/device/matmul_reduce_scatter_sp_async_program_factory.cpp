// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/matmul_reduce_scatter_sp_async/device/matmul_reduce_scatter_sp_async_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_line_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/reduce_scatter_minimal_async/device/reduce_scatter_ring_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"

namespace ttnn::experimental::prim {

// ---- Core grid split -------------------------------------------------------------------------------------------

tt::tt_metal::CoreCoord sp_matmul_core_grid(const Tensor& input, uint32_t ccl_core_rows) {
    const auto grid = input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        ccl_core_rows < grid.y,
        "ccl_core_rows ({}) must leave at least one row for the matmul (grid.y={})",
        ccl_core_rows,
        grid.y);
    return tt::tt_metal::CoreCoord(grid.x, grid.y - ccl_core_rows);
}

tt::tt_metal::CoreCoord sp_reduce_scatter_core_grid_offset(const Tensor& input, uint32_t ccl_core_rows) {
    const auto grid = input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        ccl_core_rows < grid.y,
        "ccl_core_rows ({}) must leave at least one row for the matmul (grid.y={})",
        ccl_core_rows,
        grid.y);
    return tt::tt_metal::CoreCoord(0, grid.y - ccl_core_rows);
}

uint32_t sp_reduce_scatter_core_count(ttnn::ccl::Topology topology, uint32_t num_links, uint32_t num_workers_per_link) {
    // Ring builder: a mux core per (link, direction) only when there is more than one worker; Line builder: always.
    const uint32_t mux_cores_per_direction =
        (topology == ttnn::ccl::Topology::Ring && num_workers_per_link == 1) ? 0 : 1;
    constexpr uint32_t num_directions = 2;
    return num_links * num_directions * (num_workers_per_link + mux_cores_per_direction);
}

uint32_t sp_default_reduce_scatter_workers(
    const Tensor& input, ttnn::ccl::Topology topology, uint32_t num_links, uint32_t ccl_core_rows) {
    // Measured on the 12x10 Blackhole grid with the Llama-8B TP4 shapes (see the test's perf table), fused,
    // overlapped, us/op at 2 links: Ring out_proj/w2/dgrad 375/670/389 with 5 workers vs 400/725/435 with 4;
    // Linear 408/416/717 with 4 vs 413/438/718 with 5 (and 8 is slower than both on both topologies). The
    // standalone op's data-size heuristic would pick 4 (Ring) / 8 (Linear) here.
    const uint32_t preferred = topology == ttnn::ccl::Topology::Ring ? 5 : 4;
    const auto grid = input.device()->compute_with_storage_grid_size();
    const uint32_t budget = ccl_core_rows * grid.x;
    for (uint32_t workers : {preferred, 4u, 2u, 1u}) {
        if (workers <= preferred && sp_reduce_scatter_core_count(topology, num_links, workers) <= budget) {
            return workers;
        }
    }
    return 1;
}

ttnn::prim::MatmulParams resolve_sp_matmul_params(
    const MatmulReduceScatterSpAsyncParams& args, const MatmulReduceScatterSpAsyncInputs& tensor_args) {
    ttnn::prim::MatmulParams mm = args.matmul_params;
    mm.bcast_batch = true;
    TT_FATAL(
        mm.compute_kernel_config.has_value(),
        "matmul_reduce_scatter_sp_async: matmul compute_kernel_config must be resolved");
    if (!mm.program_config.has_value()) {
        const Tensor in0_view =
            ttnn::experimental::ccl::sub_batched_view(tensor_args.input, args.reduce_scatter_params.ring_size);
        mm.program_config = ttnn::experimental::ccl::sp_matmul_program_config(
            in0_view,
            tensor_args.weight,
            sp_matmul_core_grid(tensor_args.input, args.ccl_core_rows),
            mm.transpose_b,
            mm.compute_kernel_config.value());
    }
    return mm;
}

// ---- Schedule plumbing -----------------------------------------------------------------------------------------

std::vector<ttnn::experimental::ccl::SpSubBatch> sp_matmul_schedule(
    ttnn::ccl::Topology topology, uint32_t B, uint32_t T, uint32_t ring_index) {
    const auto slice_order = rs_first_touch_order(topology, T, ring_index);
    std::vector<ttnn::experimental::ccl::SpSubBatch> schedule;
    schedule.reserve(static_cast<size_t>(B) * T);
    for (uint32_t b = 0; b < B; ++b) {
        for (uint32_t s : slice_order) {
            const uint32_t sub_batch = b * T + s;
            schedule.push_back({.in0_idx = sub_batch, .out_idx = sub_batch});
        }
    }
    return schedule;
}

std::vector<uint32_t> rs_first_touch_order(ttnn::ccl::Topology topology, uint32_t T, uint32_t r) {
    TT_FATAL(T >= 2 && r < T, "rs_first_touch_order: need T >= 2 and r < T, got T={} r={}", T, r);
    std::vector<uint32_t> order;
    order.reserve(T);
    if (topology == ttnn::ccl::Topology::Ring) {
        TT_FATAL(T % 2 == 0, "rs_first_touch_order: the Ring reduce-scatter needs an even T, got {}", T);
        const uint32_t half = T / 2;
        // Iteration 0: both direction cores read slice r + T/2 (half a slice each).
        order.push_back((r + half) % T);
        // Iterations 1..T/2-1: forward core reads r+T/2-i, backward core reads r+T/2+i.
        for (uint32_t i = 1; i < half; ++i) {
            order.push_back((r + half - i) % T);
            order.push_back((r + half + i) % T);
        }
        // Iteration T/2: both read the local slice r (final reduction).
        order.push_back(r);
    } else {
        TT_FATAL(topology == ttnn::ccl::Topology::Linear, "rs_first_touch_order: topology must be Ring or Linear");
        // FWD core: T-1, T-2, ..., r+1 (T-1-r slices); BWD core: 0, 1, ..., r-1 (r slices); interleaved, r last.
        std::vector<uint32_t> fwd;
        for (uint32_t s = T - 1; s > r; --s) {
            fwd.push_back(s);
        }
        std::vector<uint32_t> bwd;
        for (uint32_t s = 0; s < r; ++s) {
            bwd.push_back(s);
        }
        for (size_t i = 0; i < std::max(fwd.size(), bwd.size()); ++i) {
            if (i < fwd.size()) {
                order.push_back(fwd[i]);
            }
            if (i < bwd.size()) {
                order.push_back(bwd[i]);
            }
        }
        order.push_back(r);
    }
    // Sanity: a permutation of 0..T-1 ending on the local slice.
    TT_FATAL(order.size() == T, "rs_first_touch_order: produced {} entries for T={}", order.size(), T);
    std::vector<bool> seen(T, false);
    for (uint32_t s : order) {
        TT_FATAL(s < T && !seen[s], "rs_first_touch_order: slice {} repeated or out of range (T={}, r={})", s, T, r);
        seen[s] = true;
    }
    TT_FATAL(order.back() == r, "rs_first_touch_order: the local slice {} must come last (got {})", r, order.back());
    return order;
}

// ---- Program factory -------------------------------------------------------------------------------------------

MatmulReduceScatterSpAsyncProgramFactory::cached_mesh_workload_t
MatmulReduceScatterSpAsyncProgramFactory::create_mesh_workload(
    const MatmulReduceScatterSpAsyncParams& args,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const MatmulReduceScatterSpAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(args, coord, tensor_args, output_tensors);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

MatmulReduceScatterSpAsyncProgramFactory::cached_program_t MatmulReduceScatterSpAsyncProgramFactory::create_at(
    const MatmulReduceScatterSpAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const MatmulReduceScatterSpAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& rs = args.reduce_scatter_params;
    const uint32_t T = rs.ring_size;
    const uint32_t B = input.logical_shape()[0];

    Tensor& mm_partial = output_tensors.at(kMmPartialIdx);
    Tensor& rs_intermediate = output_tensors.at(kRsIntermediateIdx);
    Tensor& rs_output = output_tensors.at(kRsOutputIdx);
    const std::optional<Tensor> rs_penult_intermediate =
        output_tensors.size() > kRsPenultIdx ? std::optional<Tensor>(output_tensors.at(kRsPenultIdx)) : std::nullopt;

    tt::tt_metal::Program program{};

    // Ring position and neighbours along cluster_axis (as the standalone reduce_scatter_minimal_async does).
    const auto forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input, mesh_coord, 1, rs.topology, rs.cluster_axis);
    const auto backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input, mesh_coord, -1, rs.topology, rs.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");
    const uint32_t ring_index = ttnn::ccl::get_linearized_index_from_physical_coord(input, mesh_coord, rs.cluster_axis);

    // Core grid split.
    const auto grid = input.device()->compute_with_storage_grid_size();
    const CoreCoord rs_core_grid_offset = sp_reduce_scatter_core_grid_offset(input, args.ccl_core_rows);
    TT_FATAL(rs.num_workers_per_link.has_value(), "num_workers_per_link must be resolved");
    const uint32_t rs_cores = sp_reduce_scatter_core_count(rs.topology, rs.num_links, *rs.num_workers_per_link);
    TT_FATAL(
        rs_cores <= args.ccl_core_rows * grid.x,
        "reduce-scatter needs {} cores but only {} are reserved ({} rows x {} columns)",
        rs_cores,
        args.ccl_core_rows * grid.x,
        args.ccl_core_rows,
        grid.x);

    // 1) Reduce-scatter first: it creates the fused-op semaphore on its worker cores that the matmul signals.
    std::optional<ttnn::experimental::ccl::ReduceScatterFusedOpSignaler> reduce_scatter_fused_op_signaler =
        ttnn::experimental::ccl::ReduceScatterFusedOpSignaler();
    reduce_scatter_fused_op_signaler->init_fused_op();

    // Matmul sub-batch order = the order this rank's RS readers first touch their local slices (batch-major), and
    // the RS-side ordinal table (ordinal[b*T+s] = matmul iteration that finishes sub-batch (b, s)). The measurement
    // knob makes every RS wait resolve only after the LAST sub-batch (no overlap), isolating the cost of running the
    // RS on its reserved rows from the overlap gain.
    const auto schedule = sp_matmul_schedule(rs.topology, B, T, ring_index);
    const std::optional<std::vector<uint32_t>> sp_slice_ordinals =
        args.debug_serialize_reduce_scatter ? std::vector<uint32_t>(static_cast<size_t>(B) * T, B * T - 1)
                                            : ttnn::experimental::ccl::sp_rs_ordinals(schedule, B, T);

    // Selected through a function pointer, so every argument (incl. the schedule) is passed explicitly.
    const auto build_rs = rs.topology == ttnn::ccl::Topology::Ring
                              ? &build_ring_reduce_scatter_minimal_async_program_artifacts
                              : &build_line_reduce_scatter_minimal_async_program_artifacts;
    auto reduce_scatter_artifacts = build_rs(
        program,
        mm_partial,
        rs_intermediate,
        rs_penult_intermediate,
        mesh_coord,
        forward_coord,
        backward_coord,
        rs_output,
        rs.dim,
        rs.num_links,
        rs.ring_size,
        ring_index,
        rs.topology,
        rs.semaphore,
        rs.barrier_semaphore,
        rs.using_persistent_buffers,
        rs.sub_device_id,
        reduce_scatter_fused_op_signaler,
        rs.chunks_per_sync,
        rs.num_workers_per_link,
        rs.num_buffers_per_channel,
        rs_core_grid_offset,
        rs.compute_kernel_config,
        sp_slice_ordinals);

    // 2) Matmul on the sub-batched views in the schedule's order, signalling the RS cores once per sub-batch
    // (SP_REDUCE_SCATTER: the REDUCE_SCATTER all-matmul-core barrier + +1 on every RS worker core per iteration,
    // plus the SP_SLICE_SCHEDULE words that reorder the batch loop).
    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::SP_REDUCE_SCATTER);
    matmul_fused_op_signaler->init_reduce_scatter(
        reduce_scatter_fused_op_signaler->fused_op_receiver_cores_noc,
        reduce_scatter_fused_op_signaler->fused_op_receiver_signal_semaphores,
        reduce_scatter_fused_op_signaler->fused_op_signaler_mode);
    matmul_fused_op_signaler->init_sp_schedule(ttnn::experimental::ccl::pack_sp_schedule(schedule));
    matmul_fused_op_signaler->sp_in1_resident = args.in1_resident;

    const ttnn::prim::MatmulParams mm_params = resolve_sp_matmul_params(args, tensor_args);
    const Tensor in0_view = ttnn::experimental::ccl::sub_batched_view(input, T);
    Tensor out_view = ttnn::experimental::ccl::sub_batched_view(mm_partial, T);

    auto matmul_cached_program = ttnn::prim::matmul_multi_core_reuse_mcast_2d_optimized_helper(
        program,
        in0_view,
        weight,
        /*bias=*/std::nullopt,
        out_view,
        /*broadcast_batch=*/true,
        mm_params.compute_kernel_config.value(),
        mm_params.program_config.value(),
        /*untilize_out=*/false,
        matmul_fused_op_signaler,
        mm_params.transpose_a,
        mm_params.transpose_b);

    return cached_program_t{
        std::move(matmul_cached_program.program),
        {.reduce_scatter_artifacts = std::move(reduce_scatter_artifacts),
         .matmul_shared_variables = std::move(matmul_cached_program.shared_variables)}};
}

void MatmulReduceScatterSpAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const MatmulReduceScatterSpAsyncParams& args,
    const MatmulReduceScatterSpAsyncInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    const auto& rs = args.reduce_scatter_params;
    const uint32_t T = rs.ring_size;
    const Tensor& mm_partial = output_tensors.at(kMmPartialIdx);
    const Tensor& rs_intermediate = output_tensors.at(kRsIntermediateIdx);
    const Tensor& rs_output = output_tensors.at(kRsOutputIdx);
    const std::optional<Tensor> rs_penult_intermediate =
        output_tensors.size() > kRsPenultIdx ? std::optional<Tensor>(output_tensors.at(kRsPenultIdx)) : std::nullopt;

    // Same views as create_at (they only carry the new buffer addresses).
    const Tensor in0_view = ttnn::experimental::ccl::sub_batched_view(tensor_args.input, T);
    std::vector<Tensor> matmul_output_tensors = {ttnn::experimental::ccl::sub_batched_view(mm_partial, T)};

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
            program,
            shared_vars.matmul_shared_variables,
            args.matmul_params,
            ttnn::prim::MatmulInputs{
                .input_tensors = {in0_view, tensor_args.weight},
                .optional_input_tensors = {std::nullopt},
                .optional_output_tensors = {}},
            matmul_output_tensors);

        const auto& a = shared_vars.reduce_scatter_artifacts;
        if (rs.topology == ttnn::ccl::Topology::Ring) {
            ring_reduce_scatter_minimal_async_helper_override_runtime_arguments(
                program,
                a.reader_kernel_id,
                a.writer_kernel_id,
                a.all_cores,
                rs.num_links,
                a.num_directions_per_link,
                a.num_workers_per_direction,
                a.num_mux_cores_per_direction_per_link,
                a.num_cores_per_link,
                a.normalized_dim,
                rs.barrier_semaphore,
                rs.semaphore,
                mm_partial,
                rs_intermediate,
                rs_output,
                rs_penult_intermediate);
        } else {
            line_reduce_scatter_minimal_async_helper_override_runtime_arguments(
                program,
                a.reader_kernel_id,
                a.writer_kernel_id,
                a.all_cores,
                rs.num_links,
                a.num_directions_per_link,
                a.num_workers_per_direction,
                a.num_mux_cores_per_direction_per_link,
                a.num_cores_per_link,
                a.normalized_dim,
                rs.barrier_semaphore,
                rs.semaphore,
                mm_partial,
                rs_intermediate,
                rs_output);
        }
    }
}

}  // namespace ttnn::experimental::prim

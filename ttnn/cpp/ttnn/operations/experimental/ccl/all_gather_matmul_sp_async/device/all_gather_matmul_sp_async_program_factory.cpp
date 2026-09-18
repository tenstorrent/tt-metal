// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/all_gather_matmul_sp_async/device/all_gather_matmul_sp_async_program_factory.hpp"

#include <unordered_map>
#include <variant>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.hpp"
#include "ttnn/operations/experimental/ccl/sp_matmul_fusion_common/sp_matmul_fusion_common.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_2d_program_factory.hpp"

namespace ttnn::experimental::prim {

AllGatherMatmulSpAsyncMeshWorkloadFactory::cached_program_t AllGatherMatmulSpAsyncMeshWorkloadFactory::create_at(
    const AllGatherMatmulSpAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coord,
    const AllGatherMatmulSpAsyncInputs& tensor_args,
    AllGatherMatmulSpAsyncResult& outputs) {
    using namespace ttnn::experimental::ccl;
    const auto& ag = args.all_gather;
    const Tensor& input = tensor_args.input;
    const Tensor& weight = tensor_args.weight;
    Tensor& gathered = outputs.at(0);
    Tensor& mm = outputs.at(1);

    const uint32_t T = ag.ring_size;
    const uint32_t ring_index = ttnn::ccl::get_linearized_index_from_physical_coord(input, mesh_coord, ag.cluster_axis);
    const std::optional<MeshCoordinate> forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input, mesh_coord, 1, ag.topology, ag.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input, mesh_coord, -1, ag.topology, ag.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "forward_coord or backward_coord is null");
    const uint32_t B = input.logical_shape()[0];

    tt::tt_metal::Program program{};

    // ---- Matmul on the sub-batched views: one matmul "batch" = one (batch b, sequence slice t) ----------------
    // Local slices are read straight from the input (alt address), remote ones from the gathered output after the
    // all-gather's per-direction semaphore says they landed (SP_AG_WAIT in the in0 sender kernel).
    const Tensor gathered_view = sub_batched_view(gathered, T);
    Tensor mm_view = sub_batched_view(mm, T);
    std::optional<MatmulFusedOpSignaler> matmul_fused_op_signaler =
        MatmulFusedOpSignaler(MatmulFusedOpSignalerType::SP_ALL_GATHER);
    const auto schedule = args.debug_serialize_ag ? sp_ag_schedule_serialized(ag.topology, T, ring_index, B)
                                                  : sp_ag_schedule(ag.topology, T, ring_index, B);
    matmul_fused_op_signaler->init_sp_schedule(
        pack_sp_schedule(schedule), static_cast<uint32_t>(input.buffer()->address()));
    matmul_fused_op_signaler->sp_in1_resident = args.in1_resident;

    auto matmul_cached_program = ttnn::prim::matmul_multi_core_reuse_mcast_2d_optimized_helper(
        program,
        gathered_view,
        weight,
        tensor_args.bias,
        mm_view,
        args.matmul.bcast_batch.value(),
        args.matmul.compute_kernel_config.value(),
        args.matmul.program_config.value(),
        args.matmul.untilize_out,
        matmul_fused_op_signaler,
        args.matmul.transpose_a,
        args.matmul.transpose_b);
    program = std::move(matmul_cached_program.program);
    auto matmul_shared_variables = std::move(matmul_cached_program.shared_variables);
    TT_FATAL(
        matmul_shared_variables.sp_in0_alt_addr_rt_arg_idx != 0,
        "all_gather_matmul_sp_async: the matmul program did not record the in0 alternate-address rt arg");

    // ---- All-gather, signalling the matmul's in0 sender cores (MULTI mode, 2 direction semaphores) -------------
    std::optional<AllGatherFusedOpSignaler> all_gather_fused_op_signaler = AllGatherFusedOpSignaler();
    all_gather_fused_op_signaler->init_fused_op(
        matmul_fused_op_signaler->fused_op_receiver_cores_noc,
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores,
        matmul_fused_op_signaler->fused_op_signaler_mode);

    auto all_gather_artifacts = ttnn::build_all_gather_async_minimal_default_program_artifacts(
        program,
        input,
        mesh_coord,
        forward_coord,
        backward_coord,
        gathered,
        ag.dim,
        ag.num_links,
        T,
        ring_index,
        ag.topology,
        ag.semaphore,
        ag.barrier_semaphore,
        ag.using_persistent_buffers,
        ag.sub_device_id,
        all_gather_fused_op_signaler,
        ag.chunks_per_sync,
        ag.num_workers_per_link,
        ag.num_buffers_per_channel,
        args.all_gather_core_grid_offset,
        /*reverse_order=*/false,
        /*sub_core_grid=*/std::nullopt,
        /*fused_op_signal_on_receive=*/args.ag_signal_on_receive);

    return cached_program_t(
        {std::move(program),
         shared_variables_t{
             .matmul = std::move(matmul_shared_variables), .all_gather = std::move(all_gather_artifacts)}});
}

AllGatherMatmulSpAsyncMeshWorkloadFactory::cached_mesh_workload_t
AllGatherMatmulSpAsyncMeshWorkloadFactory::create_mesh_workload(
    const AllGatherMatmulSpAsyncParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllGatherMatmulSpAsyncInputs& tensor_args,
    AllGatherMatmulSpAsyncResult& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& mesh_coord : tensor_coords.coords()) {
        const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
        auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
        workload.add_program(single_coord_range, std::move(cached_program.program));
        shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
    }
    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void AllGatherMatmulSpAsyncMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllGatherMatmulSpAsyncParams& operation_attributes,
    const AllGatherMatmulSpAsyncInputs& tensor_args,
    AllGatherMatmulSpAsyncResult& tensor_return_value) {
    const auto& ag = operation_attributes.all_gather;
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        // Matmul: gathered (in0), weight, bias, mm output. Only addresses are read, so the un-viewed tensors serve.
        std::vector<Tensor> matmul_output_tensors = {tensor_return_value[1]};
        ttnn::prim::MatmulMultiCoreReuseMcast2DProgramFactory::override_runtime_arguments(
            program,
            shared_vars.matmul,
            operation_attributes.matmul,
            {.input_tensors = {tensor_return_value[0], tensor_args.weight},
             .optional_input_tensors = {tensor_args.bias},
             .optional_output_tensors = {tensor_return_value[1]}},
            matmul_output_tensors);
        // The matmul's own override does not know about the alternate in0 buffer (the sharded input).
        ttnn::prim::override_sp_in0_alt_addr(
            program, shared_vars.matmul, static_cast<uint32_t>(tensor_args.input.buffer()->address()));

        const auto& a = shared_vars.all_gather;
        all_gather_async_minimal_default_helper_override_runtime_arguments(
            program,
            a.reader_kernel_id,
            a.writer_kernel_id,
            a.all_cores,
            ag.num_links,
            a.num_directions_per_link,
            a.num_workers_per_direction,
            a.num_mux_cores_per_direction_per_link,
            a.num_cores_per_link,
            ag.barrier_semaphore,
            ag.semaphore,
            tensor_args.input,
            tensor_return_value[0]);
    }
}

}  // namespace ttnn::experimental::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/sparse/factory/sparse_matmul_multicore_reuse_mcast_1d_spec_factory.hpp"

#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts SparseMatmulMultiCoreReuseMcast1DSpecFactory::create_program_artifacts(
    const SparseMatmulParams& operation_attributes,
    const SparseMatmulInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    using namespace operations::matmul::utilities;

    const auto& a = tensor_args.input_tensors.at(0);
    const auto& b = tensor_args.input_tensors.at(1);
    const auto& sparsity = tensor_args.input_tensors.at(2);
    const auto& output = tensor_return_value.at(0);

    // The program config is resolved exactly as the ProgramDescriptor factory resolves it.
    const auto matmul_attributes = MatmulParams{
        operation_attributes.program_config,
        /*bcast_batch=*/std::nullopt,
        operation_attributes.output_mem_config,
        operation_attributes.output_dtype,
        operation_attributes.compute_kernel_config,
        /*untilize_out=*/false,
        operation_attributes.user_core_coord,
        /*user_fused_activation=*/std::nullopt,
        /*user_run_batched=*/false,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        operation_attributes.output_tile,
        operation_attributes.global_cb,
        operation_attributes.sub_device_id};
    auto chosen_program_config = operations::matmul::get_program_config(
        a, b, /*transpose_a=*/false, /*transpose_b=*/false, /*bias_single_tile_size=*/0, matmul_attributes);
    operations::matmul::normalize_program_config(chosen_program_config, a.device()->compute_with_storage_grid_size());
    const auto program_config =
        std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);

    // When both inputs are sparse they share the group axis; when only B is, A's batch dims are an
    // outer loop over every group. Validation restricts the outer loop to one sparsity page.
    const auto& ashape = get_matmul_tensor_padded_shape(a, /*transpose=*/false);
    const auto& bshape = get_matmul_tensor_padded_shape(b, /*transpose=*/false);
    const uint32_t batchB = get_batch_size(bshape);
    const uint32_t batchA = operation_attributes.is_input_a_sparse ? 1u : get_batch_size(ashape);

    const auto nnz = operation_attributes.nnz;
    // Compact output packs only the nnz active groups, in mask order. Detected as the device op and the
    // ProgramDescriptor factory detect it: by shape, [1, nnz, M, N].
    const bool compact_output =
        nnz.has_value() &&
        output.logical_shape() == ttnn::Shape{1U, nnz.value(), a.logical_shape()[-2], b.logical_shape()[-1]};
    const McastIn0Sparsity sparsity_args{
        .mask = sparsity.mesh_tensor(),
        .batchB = batchB,
        .num_batch_compute = nnz.value_or(static_cast<uint32_t>(sparsity.logical_volume())),
        .get_batch_from_reader = !nnz.has_value(),
        .bcast_A = !operation_attributes.is_input_a_sparse,
        .compact_output = compact_output,
    };

    return reuse_mcast_1d_optimized_helpers::create_sparse_mcast_in0_artifacts(
        a,
        b,
        output,
        program_config,
        operation_attributes.compute_kernel_config.value(),
        batchA,
        operation_attributes.prefetcher_pipes,
        sparsity_args);
}

}  // namespace ttnn::prim

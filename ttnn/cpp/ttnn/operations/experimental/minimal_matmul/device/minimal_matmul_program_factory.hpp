// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "minimal_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"

namespace ttnn::experimental::prim {

// LEGACY, CCL-ONLY. minimal_matmul itself no longer uses this: MinimalMatmulDeviceOperation is on
// the ProgramDescriptor path (minimal_matmul_program_descriptor.cpp).
//
// It survives because minimal_matmul_strided_reduce_scatter_async builds one fused Program holding
// both the reduce-scatter and the matmul kernels: it calls minimal_matmul_factory_helper_common
// with its own Program&, stores the returned shared_variables_t, and re-enters
// override_runtime_arguments through cached_program_t::proxy.
//
struct MinimalMatmulProgramFactory {
    struct shared_variables_t {
        uint32_t num_cores{};
        std::vector<CoreCoord> cores;
        tt::tt_metal::KernelHandle in0_sender_kernels_id{};
        tt::tt_metal::KernelHandle in0_receiver_kernels_id{};
        tt::tt_metal::KernelHandle in1_sender_kernels_id{};
        tt::tt_metal::KernelHandle in1_receiver_kernels_id{};
        tt::tt_metal::KernelHandle compute_kernels_id{};
        bool transpose_core_grid{};
        bool read_local_slice_from_input{};
        // Fused concatenation: in0's K is sourced from input_tensor + optional_input_tensor (no
        // materialized concat). When set, optional_input_tensor feeds the in3 address on re-runs.
        bool two_input_split{};
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const MinimalMatmulParams& operation_attributes,
        const MinimalMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

// Builds the matmul kernels into an existing (typically fused) Program. Takes a number of output
// tensors (N_chunks) and a vector of output tensors, so it serves both the single-output and the
// minimal_matmul_split shapes.
MinimalMatmulProgramFactory::shared_variables_t minimal_matmul_factory_helper_common(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    const Tensor& weight_tensor,
    const std::optional<const Tensor>& bias_tensor,
    const std::optional<operations::unary::UnaryWithParam>& fused_activation,
    const std::optional<const MinimalMatmulConfig>& config,
    const std::vector<Tensor>& output_tensors,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler>& fused_op_signaler,
    uint32_t N_chunks,
    std::optional<float> fused_ternary_scalar = std::nullopt,
    const std::optional<const Tensor>& fused_ternary_input_a = std::nullopt,
    const std::optional<const Tensor>& fused_ternary_input_b = std::nullopt,
    std::optional<ttnn::experimental::ccl::StridedReduceScatterFusedOpSignaler> srs_fused_op_signaler = std::nullopt,
    bool fuse_swiglu = false,
    // Fused concat (concat-free): when set, in0's K is sourced from input_tensor (prefix tiles) then
    // optional_input_tensor (suffix tiles). The split point is input_tensor's own K width.
    const std::optional<const Tensor>& optional_input_tensor = std::nullopt);

}  // namespace ttnn::experimental::prim

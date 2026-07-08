// SPDX-FileCopyrightText: (C) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/transformer/concat_heads_matmul/device/concat_heads_matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_1d_program_factory.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

// Wraps the stock tuned 1D-mcast matmul factory. attn is view'd to [1,1,seq,K] at host/build time
// only (the view never becomes a traced op; the captured program reads attn's buffer, and attn is
// this op's declared input -> trace-replay-safe).
struct ConcatHeadsMatmulSharedVariables {
    ttnn::prim::matmul_mcast_1d_common_override_variables_t mm_shared;
    ttnn::operations::matmul::MatmulProgramConfig program_config;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ConcatHeadsMatmulProgramFactory {
    using shared_variables_t = ConcatHeadsMatmulSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ConcatHeadsMatmulParams& operation_attributes,
        const ConcatHeadsMatmulInputs& tensor_args,
        Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ConcatHeadsMatmulParams& operation_attributes,
        const ConcatHeadsMatmulInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim

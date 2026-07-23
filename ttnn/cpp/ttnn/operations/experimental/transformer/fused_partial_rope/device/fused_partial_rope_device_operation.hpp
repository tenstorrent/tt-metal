// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::transformer::fused_partial_rope {

// -----------------------------------------------------------------------------
// FusedPartialRopeDeviceOperation
//
// Fuses the deepseek_v4_flash `_apply_rope` calc into one device op: interleaved
// RoPE on the trailing `rope_dim` channels of a height-sharded `[1, 1, rows, D]`
// input, with the leading (D - rope_dim) "nope" channels passed through
// untouched. The rotation uses the `rope_dim`-wide `rotate_half` matmul form:
//
//   out[..., :D-Rd] = x[..., :D-Rd]
//   out[..., D-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin
//
// Layout (v1): height-sharded L1, one tile-row (32 rows) per core, so
// num_cores = ceil(rows / 32). `cos`/`sin` are `[1, 1, rows, Rd]` sharded on the
// same core grid; `trans_mat` is a single [32, 32] rotate_half tile, replicated.
// -----------------------------------------------------------------------------
struct FusedPartialRopeDeviceOperation {
    struct operation_attributes_t {
        uint32_t rope_dim;
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& trans_mat;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ShardedProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::fused_partial_rope

namespace ttnn::prim {

ttnn::Tensor fused_partial_rope(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t rope_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config);

}  // namespace ttnn::prim

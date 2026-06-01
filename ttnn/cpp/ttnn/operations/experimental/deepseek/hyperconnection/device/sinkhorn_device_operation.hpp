// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "sinkhorn_program_factory.hpp"
#include "sinkhorn_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct SinkhornDeviceOperation {
    using operation_attributes_t = SinkhornParams;
    using tensor_args_t = SinkhornInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = SinkhornTensorReturn;
    using program_factory_t = std::variant<SinkhornProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection

namespace ttnn::prim {

// Decode-only (T == 1) comb / Sinkhorn-Knopp stage. Given the reshaped comb projection and bias
// (each [1,1,H,H], a single tile), computes:
//   comb = softmax(comb_w * comb_scale + comb_bias, dim=-1) + eps
//   comb = comb / (sum(comb, dim=-2) + eps)                          (initial column pass)
//   repeat sinkhorn_iters-1 times: row pass (dim=-1) then column pass (dim=-2)
// Returns comb [1,1,H,H].
Tensor fused_hyperconnection_sinkhorn(
    const Tensor& comb_w,
    const Tensor& comb_bias,
    uint32_t num_streams,
    uint32_t sinkhorn_iters,
    float comb_scale,
    float eps,
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::prim

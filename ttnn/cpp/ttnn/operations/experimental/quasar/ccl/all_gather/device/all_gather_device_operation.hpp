// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "all_gather_device_operation_types.hpp"
#include "all_gather_program_factory.hpp"

namespace ttnn::prim::qsr {

// Intra-chip all-gather on Quasar.
//
// Input: a TILE tensor height- or width-sharded over N clusters, one shard per cluster. Rank r is the
// cluster holding shard r, and shard r is rank r's slice. The gather dim is the sharding dim.
// Output: [N, ...input shape...], height-sharded over the same N clusters with shard shape
// [prod(leading dims) * H, W], so output shard r is the full gathered tensor: out[r] == input for every r.
// This is what the Quasar matmul in0 ring leaves in cb_in2 on every core, as a first-class tensor.
struct AllGatherDeviceOperation {
    using operation_attributes_t = AllGatherOperationAttributes;
    using tensor_args_t = AllGatherTensorArgs;
    using tensor_return_value_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;

    using program_factory_t = std::variant<AllGatherProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

Tensor all_gather(const Tensor& input_tensor, const AllGatherOperationAttributes& attributes);

}  // namespace ttnn::prim::qsr

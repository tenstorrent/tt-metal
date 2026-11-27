// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/program_descriptors.hpp>
// Reuse the router-affinity activation enum from moe_grouped_topk (no duplication).
#include "ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/moe_grouped_topk_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate {

using ScoreFunc = ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::ScoreFunc;

struct MoeHashGateDeviceOperation {
    struct operation_attributes_t {
        uint32_t n_activated_experts;
        float route_scale;
        float epsilon;
        ScoreFunc score_func;
        tt::tt_metal::MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& scores;
        const Tensor& input_ids;
        const Tensor& tid2eid;
        std::optional<Tensor> padding_config;
    };

    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::MoeHashGateDeviceOperation::tensor_return_value_t
moe_hash_gate(
    const Tensor& scores,
    const Tensor& input_ids,
    const Tensor& tid2eid,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::ScoreFunc score_func =
        ttnn::operations::experimental::deepseek_prefill::moe_hash_gate::ScoreFunc::SqrtSoftplus,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& padding_config = std::nullopt);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

// Router affinity activation applied to the gate logits before bias-add / top-k.
// Sigmoid is DeepSeek-V3 / Kimi; SqrtSoftplus (== sqrt(softplus(x))) is DeepSeek-V4.
enum class ScoreFunc : uint32_t {
    Sigmoid = 0,
    SqrtSoftplus = 1,
};

struct MoeGroupedTopkDeviceOperation {
    struct operation_attributes_t {
        uint32_t n_groups;
        uint32_t summed_experts_per_group;
        uint32_t topk_groups;
        uint32_t n_activated_experts;
        float route_scale;
        float epsilon;
        bool stable_sort;
        ScoreFunc score_func;
        tt::tt_metal::MemoryConfig output_mem_config;
    };

    struct tensor_args_t {
        const Tensor& scores;
        const Tensor& bias;
        std::optional<Tensor> padding_config;
    };

    using spec_return_value_t = std::array<tt::tt_metal::TensorSpec, 2>;
    using tensor_return_value_t = std::array<Tensor, 2>;

    struct ProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle reader_kernel_id{};
            tt::tt_metal::KernelHandle writer_kernel_id{};
            tt::tt_metal::KernelHandle compute_kernel_id{};
            std::vector<tt::tt_metal::CoreCoord> cores;
        };

        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
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

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk

namespace ttnn::prim {

ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::MoeGroupedTopkDeviceOperation::tensor_return_value_t
moe_grouped_topk(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    bool stable_sort = false,
    ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::ScoreFunc score_func =
        ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk::ScoreFunc::Sigmoid,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config = std::nullopt,
    const std::optional<Tensor>& padding_config = std::nullopt);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk.hpp"

#include <cstdint>
#include <string>

#include <tt_stl/assert.hpp>
#include "device/moe_grouped_topk_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

namespace {
ScoreFunc parse_score_func(const std::string& score_func) {
    if (score_func == "sigmoid") {
        return ScoreFunc::Sigmoid;
    }
    if (score_func == "sqrtsoftplus") {
        return ScoreFunc::SqrtSoftplus;
    }
    TT_THROW("Unsupported score_func '{}'. Expected 'sigmoid' or 'sqrtsoftplus'.", score_func);
}
}  // namespace

std::array<Tensor, 2> moe_grouped_topk(
    const Tensor& scores,
    const Tensor& bias,
    uint32_t n_groups,
    uint32_t summed_experts_per_group,
    uint32_t topk_groups,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    bool stable_sort,
    const std::string& score_func,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& padding_config) {
    return ttnn::prim::moe_grouped_topk(
        scores,
        bias,
        n_groups,
        summed_experts_per_group,
        topk_groups,
        n_activated_experts,
        route_scale,
        epsilon,
        stable_sort,
        parse_score_func(score_func),
        output_mem_config,
        padding_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk

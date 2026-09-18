// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_hash_gate.hpp"

#include <cstdint>
#include <string>

#include <tt_stl/assert.hpp>
#include "device/moe_hash_gate_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate {

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

std::array<Tensor, 2> moe_hash_gate(
    const Tensor& scores,
    const Tensor& input_ids,
    const Tensor& tid2eid,
    uint32_t n_activated_experts,
    float route_scale,
    float epsilon,
    const std::string& score_func,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& padding_config) {
    return ttnn::prim::moe_hash_gate(
        scores,
        input_ids,
        tid2eid,
        n_activated_experts,
        route_scale,
        epsilon,
        parse_score_func(score_func),
        output_mem_config,
        padding_config);
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate

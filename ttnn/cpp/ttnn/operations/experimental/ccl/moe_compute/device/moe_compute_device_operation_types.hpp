// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/base_types.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct MoEComputeParams {
    uint32_t layer_id;
    std::optional<uint32_t> cluster_axis;

    auto attributes() const {
        using tt::stl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.emplace_back("layer_id", layer_id);
        attrs.emplace_back("cluster_axis", cluster_axis);
        return attrs;
    }
};

struct MoEComputeInputs {
    const ttnn::Tensor& tilize_input_tensor;
    const ttnn::Tensor& tilize_expert_indices_tensor;
    const ttnn::Tensor& tilize_expert_scores_tensor;
    const ttnn::Tensor& tilize_expert_mapping_tensor;
    const ttnn::Tensor& matmul_w0_w1_tensor;
    const ttnn::Tensor& matmul_w2_tensor;
};

}  // namespace ttnn::experimental::prim

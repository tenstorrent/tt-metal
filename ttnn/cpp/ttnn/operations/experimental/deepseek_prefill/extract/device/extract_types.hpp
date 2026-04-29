// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::extract {

struct ExtractParams {
    uint32_t global_expert_id;
    uint32_t max_dispatched_tokens_per_expert;

    static constexpr auto attribute_names =
        std::forward_as_tuple("global_expert_id", "max_dispatched_tokens_per_expert");

    auto attribute_values() const { return std::forward_as_tuple(global_expert_id, max_dispatched_tokens_per_expert); }
};

struct ExtractInputs {
    Tensor global_tensor;
    Tensor start;
    Tensor counts;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::extract

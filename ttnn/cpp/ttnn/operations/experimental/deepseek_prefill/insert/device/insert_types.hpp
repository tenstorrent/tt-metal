// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::insert {

struct InsertParams {
    uint32_t global_expert_id;

    static constexpr auto attribute_names = std::forward_as_tuple("global_expert_id");

    auto attribute_values() const { return std::forward_as_tuple(global_expert_id); }
};

struct InsertInputs {
    Tensor global_tensor;
    Tensor local_tensor;
    Tensor start;
    Tensor counts;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::insert

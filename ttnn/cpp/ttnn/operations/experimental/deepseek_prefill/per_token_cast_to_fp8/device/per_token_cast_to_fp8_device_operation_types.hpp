// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::per_token_cast_to_fp8 {

struct PerTokenCastToFp8Params {
    tt::tt_metal::MemoryConfig output_memory_config;
    bool round_scale_to_power_of_two;

    static constexpr auto attribute_names =
        std::forward_as_tuple("output_memory_config", "round_scale_to_power_of_two");
    auto attribute_values() const { return std::forward_as_tuple(output_memory_config, round_scale_to_power_of_two); }
};

struct PerTokenCastToFp8Inputs {
    const Tensor& input_tensor;
};

}  // namespace ttnn::experimental::prim::per_token_cast_to_fp8

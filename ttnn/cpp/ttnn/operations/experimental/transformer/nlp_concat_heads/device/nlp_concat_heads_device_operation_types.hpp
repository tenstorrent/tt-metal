// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct NlpConcatHeadsParams {
    tt::tt_metal::MemoryConfig output_mem_config;

    static constexpr auto attribute_names = std::forward_as_tuple("output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(output_mem_config); }
};

}  // namespace ttnn::experimental::prim

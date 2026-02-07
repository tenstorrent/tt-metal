// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::prim {

struct RepeatParams {
    uint32_t m_num_repeats{};
    bool m_is_last_dim{};
    tt::tt_metal::MemoryConfig m_output_mem_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("m_num_repeats", "m_is_last_dim", "m_output_mem_config");
    auto attribute_values() const { return std::forward_as_tuple(m_num_repeats, m_is_last_dim, m_output_mem_config); }
};

struct RepeatInputs {
    Tensor input;

    static constexpr auto attribute_names = std::forward_as_tuple("input");
    auto attribute_values() const { return std::forward_as_tuple(input); }
};

}  // namespace ttnn::prim

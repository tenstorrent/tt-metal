// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::data_movement::repeat {

struct operation_attributes_t {
    uint32_t m_num_repeats{};
    bool m_is_last_dim{};
    tt::tt_metal::MemoryConfig m_output_mem_config;
};

struct tensor_args_t {
    Tensor input;
};

}  // namespace ttnn::operations::data_movement::repeat

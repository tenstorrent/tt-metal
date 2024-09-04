// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"


namespace ttnn {
namespace operations::data_movement {

struct SplitOperation {
    static std::vector<ttnn::Tensor> invoke(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        int64_t& num_splits,
        int64_t& dim,
        const std::optional<MemoryConfig>& memory_config_arg);

    static std::vector<ttnn::Tensor> invoke(
        const ttnn::Tensor& input_tensor,
        int64_t& num_splits,
        int64_t& dim,
        const std::optional<MemoryConfig>& memory_config);

    static std::vector<ttnn::Tensor> invoke(const ttnn::Tensor& input_tensor, int64_t& num_splits, int64_t& dim);
};


}  // namespace operations::data_movement

constexpr auto split = ttnn::register_operation<"ttnn::split", ttnn::operations::data_movement::SplitOperation>();

}  // namespace ttnn

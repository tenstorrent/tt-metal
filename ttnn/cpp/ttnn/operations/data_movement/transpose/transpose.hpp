// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/transpose/transpose_op.hpp"


namespace ttnn {
namespace operations::data_movement {

struct ExecuteTranspose {
    static inline ttnn::Tensor operator()(
        uint8_t queue_id,
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<MemoryConfig>& memory_config) {
        return tt::tt_metal::transpose(
            (tt::tt_metal::Tensor)input_tensor, dim1, dim2, memory_config.value_or(input_tensor.memory_config()));
    }

    static inline ttnn::Tensor operator()(
        const ttnn::Tensor& input_tensor,
        const int64_t& dim1,
        const int64_t& dim2,
        const std::optional<MemoryConfig>& memory_config) {
        return operator()(DefaultQueueId, input_tensor, dim1, dim2, memory_config);
    }

    static inline ttnn::Tensor operator()(const ttnn::Tensor& input_tensor, const int64_t& dim1, const int64_t& dim2) {
        return operator()(DefaultQueueId, input_tensor, dim1, dim2, std::nullopt);
    }
};

}  // namespace operations::data_movement

constexpr auto transpose = ttnn::register_operation<"ttnn::transpose", ttnn::operations::data_movement::ExecuteTranspose>();

}  // namespace ttnn

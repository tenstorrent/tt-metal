// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm.hpp"
#include "device/fill_rm_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

ttnn::Tensor FillRMOperation::invoke(
    QueueId queue_id,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const ttnn::Tensor& any,
    float val_hi,
    float val_lo,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(
               FillRM{N, C, H, W, hFill, wFill, val_hi, val_lo, output_memory_config}, {any}, {}, {}, queue_id)
        .at(0);
}

ttnn::Tensor FillOnesRMOperation::invoke(
    QueueId queue_id,
    uint32_t N,
    uint32_t C,
    uint32_t H,
    uint32_t W,
    uint32_t hFill,
    uint32_t wFill,
    const ttnn::Tensor& any,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(
               FillRM{N, C, H, W, hFill, wFill, 1.0f, 0.0f, output_memory_config}, {any}, {}, {}, queue_id)
        .at(0);
}

}  // namespace ttnn::operations::data_movement

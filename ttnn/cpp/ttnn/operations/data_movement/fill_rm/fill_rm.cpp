// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_rm.hpp"
#include "device/fill_rm_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::data_movement{

ttnn::Tensor FillRMOperation::invoke(uint8_t queue_id, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, float val_hi, float val_lo, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(FillRM{N, C, H, W, hFill, wFill, val_hi, val_lo, output_memory_config}, {any}, {}, {}, queue_id).at(0);
}

ttnn::Tensor FillRMOperation::invoke(uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, float val_hi, float val_lo, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, N, C, H, W, hFill, wFill, any, val_hi, val_lo, memory_config_arg);
}

ttnn::Tensor FillOnesRMOperation::invoke(uint8_t queue_id, uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    return operation::run_without_autoformat(FillRM{N, C, H, W, hFill, wFill, 1.0f, 0.0f, output_memory_config},  {any}, {}, {}, queue_id).at(0);
}

ttnn::Tensor FillOnesRMOperation::invoke(uint32_t N, uint32_t C, uint32_t H, uint32_t W, uint32_t hFill, uint32_t wFill, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, N, C, H, W, hFill, wFill, any, memory_config_arg);
}

ttnn::Tensor FullOperation::invoke(uint8_t queue_id, uint32_t N, uint32_t C, uint32_t H, uint32_t W, float fill_value, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(any.memory_config());
    // Pass hFill = 0 and wFill = 0 to fill the entire tensor with `fill_value`
    return operation::run_without_autoformat(FillRM{N, C, H, W, 0, 0, fill_value, fill_value, output_memory_config}, {any}, {}, {}, queue_id).at(0);
}

ttnn::Tensor FullOperation::invoke(uint32_t N, uint32_t C, uint32_t H, uint32_t W, float fill_value, const ttnn::Tensor& any, const std::optional<ttnn::MemoryConfig>& memory_config_arg) {
    return invoke(DefaultQueueId, N, C, H, W, fill_value, any, memory_config_arg);
}

}  // namespace ttnn::operations::data_movement

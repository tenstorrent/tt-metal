// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include <tt-metalium/sub_device_types.hpp>

namespace ttnn {
namespace operations::experimental::deepseek::prefill_combine {

struct ExecutePrefillCombine {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& dispatched_tensor,
        const ttnn::Tensor& metadata_tensor,
        const ttnn::Tensor& experts_counter_tensor,
        uint32_t num_chips,
        uint32_t experts_per_chip,
        uint32_t num_experts_per_tok,
        uint32_t seq_len_per_chip,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt);
};

}  // namespace operations::experimental::deepseek::prefill_combine

constexpr auto prefill_combine = ttnn::register_operation<
    "ttnn::experimental::deepseek::prefill_combine",
    ttnn::operations::experimental::deepseek::prefill_combine::ExecutePrefillCombine>();

}  // namespace ttnn

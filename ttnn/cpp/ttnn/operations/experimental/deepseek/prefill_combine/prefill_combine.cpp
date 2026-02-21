// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_combine.hpp"
#include "device/prefill_combine_device_operation.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_combine {

ttnn::Tensor ExecutePrefillCombine::invoke(
    const ttnn::Tensor& /*dispatched_tensor*/,
    const ttnn::Tensor& /*metadata_tensor*/,
    const ttnn::Tensor& /*experts_counter_tensor*/,
    uint32_t /*num_chips*/,
    uint32_t /*experts_per_chip*/,
    uint32_t /*num_experts_per_tok*/,
    uint32_t /*seq_len_per_chip*/,
    const std::optional<ttnn::MemoryConfig>& /*memory_config*/,
    const std::optional<tt::tt_metal::SubDeviceId>& /*subdevice_id*/) {
    // TODO: Implement parameter validation
    // TODO: Get device and subdevice info
    // TODO: Call device operation

    // Stub implementation - return empty tensor
    TT_THROW("prefill_combine operation not yet implemented");
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_combine

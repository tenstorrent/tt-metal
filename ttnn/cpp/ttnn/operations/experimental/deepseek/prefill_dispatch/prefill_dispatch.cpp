// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch.hpp"
#include "device/prefill_dispatch_device_operation.hpp"
#include "ttnn/operation.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::experimental::deepseek::prefill_dispatch {

std::array<ttnn::Tensor, 3> ExecutePrefillDispatch::invoke(
    const ttnn::Tensor& /*input_tensor*/,
    const ttnn::Tensor& /*weights_tensor*/,
    const ttnn::Tensor& /*indices_tensor*/,
    uint32_t /*num_chips*/,
    uint32_t /*experts_per_chip*/,
    uint32_t /*n_routed_experts*/,
    uint32_t /*metadata_len*/,
    uint32_t /*max_dispatched_tokens_per_expert*/,
    const std::optional<ttnn::MemoryConfig>& /*memory_config*/,
    const std::optional<tt::tt_metal::SubDeviceId>& /*subdevice_id*/) {
    // TODO: Implement parameter validation
    // TODO: Get device and subdevice info
    // TODO: Call device operation

    // Stub implementation - return empty tensors
    TT_THROW("prefill_dispatch operation not yet implemented");
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch

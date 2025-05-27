// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/common/queue_id.hpp"

#include <tt-metalium/constants.hpp>

#include "multidevice_scatter.hpp"
#include "device/multidevice_scatter_device_operation.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/sub_device.hpp>

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteMultideviceScatter::invoke(
    QueueId queue_id,
    const ttnn::Tensor& input_tensor,
    const int32_t dim,
    const uint32_t cluster_axis,
    const std::optional<ttnn::MemoryConfig>& memory_config) {
    return ttnn::prim::multidevice_scatter(
        input_tensor, dim, cluster_axis, memory_config.value_or(input_tensor.memory_config()));
}

}  // namespace ttnn::operations::ccl

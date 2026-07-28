// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_device.hpp"

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor to_device(
    const ttnn::Tensor& tensor,
    MeshDevice* mesh_device,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<QueueId> queue_id) {
    auto mem_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    return tensor.to_device(mesh_device, mem_config, queue_id);
}

}  // namespace ttnn::operations::experimental::quasar

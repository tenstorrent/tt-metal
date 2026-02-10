// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_mla.hpp"
#include "device/distributed_mla_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer::sdpa_prefill {

ttnn::Tensor ExecuteDistributedMLA::invoke(
    const ttnn::Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<MemoryConfig>& memory_config) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    return ttnn::prim::distributed_mla(input_tensor, cluster_axis, output_memory_config);
}

}  // namespace ttnn::operations::transformer::sdpa_prefill

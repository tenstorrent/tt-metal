// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "to_memory_config_op.hpp"

#include "ttnn/core.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// TODO(nuked-op reshard / interleaved_to_sharded / sharded_to_interleaved):
// The original implementation dispatched to those three ops based on a set of
// can_use_* helpers. With those ops removed, fall back to ttnn::prim::copy for
// every non-trivial conversion. Consumers that depended on shard-aware paths
// will be functionally degraded until the ops are restored.
Tensor to_memory_config(
    const Tensor& tensor,
    const MemoryConfig& memory_config,
    std::optional<DataType> dtype,
    const std::optional<Tensor>& output_tensor) {
    const auto original_memory_config = ttnn::get_memory_config(tensor);
    if (original_memory_config.has_value() && original_memory_config.value() == memory_config &&
        !output_tensor.has_value()) {
        return tensor;
    }

    return ttnn::prim::copy(tensor, memory_config, dtype.value_or(tensor.dtype()), output_tensor);
}

}  // namespace ttnn

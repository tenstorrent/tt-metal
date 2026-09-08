// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "width_to_height_shard.hpp"
#include "device/width_to_height_shard_device_operation.hpp"

namespace ttnn::experimental::deepseek {

ttnn::Tensor width_to_height_shard(
    const ttnn::Tensor& input_tensor,
    const CoreRangeSet& output_core_range_set,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::width_to_height_shard(input_tensor, output_core_range_set, optional_output_tensor);
}

}  // namespace ttnn::experimental::deepseek

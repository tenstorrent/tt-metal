// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <ttnn/tensor/tensor_spec.hpp>

#include <optional>
#include <string_view>

namespace tt::tt_metal {
class Tensor;
}
namespace ttnn::prim {

struct ReduceOpDeviceGridValidationOptions {
    bool num_cores_use_last_core_divider = false;

    const tt::tt_metal::CoreRangeSet* sub_grid_contained_in_device_grid = nullptr;
    std::string_view sub_grid_label = {};

    const tt::tt_metal::MemoryConfig* shard_grid_contained_in_device_grid = nullptr;
    std::string_view memory_config_label = {};
};

void validate_reduce_op_tensor(
    const tt::tt_metal::Tensor& tensor_ref,
    std::string_view op_name,
    std::string_view tensor_label,
    const ReduceOpDeviceGridValidationOptions* core_grids_within_device_grid = nullptr,
    std::optional<tt::tt_metal::TensorSpec> tensor_spec_ref = std::nullopt);

}  // namespace ttnn::prim

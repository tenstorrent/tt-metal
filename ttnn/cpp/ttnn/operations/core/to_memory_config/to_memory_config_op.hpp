// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>

#include "ttnn/core.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/sharded/reshard/reshard.hpp"
#include "ttnn/operations/data_movement/sharded/interleaved_to_sharded/interleaved_to_sharded.hpp"
#include "ttnn/operations/data_movement/sharded/sharded_to_interleaved/device/sharded_to_interleaved_device_operation.hpp"

#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"

namespace ttnn {

Tensor to_memory_config(
    const Tensor& tensor,
    const MemoryConfig& memory_config,
    std::optional<DataType> dtype = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

// Typecast a device tensor to dtype while keeping its layout and memory config. Shared by the dtype-only
// shortcuts of to_layout and to_memory_config. op_name prefixes the error raised for a block-float dtype
// on a ROW_MAJOR tensor; sub_core_grids is forwarded to typecast for TILE tensors.
Tensor typecast_keeping_memory_config(
    const Tensor& tensor,
    DataType dtype,
    std::string_view op_name,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

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

}  // namespace ttnn

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn {

ttnn::Tensor to_sharded(
    const ttnn::Tensor& input_tensor,
    const MemoryConfig& sharded_memory_config,
    const std::optional<DataType>& data_type_arg,
    // const std::optional<bool>& keep_l1_aligned = std::nullopt,
    const std::optional<Tensor>& preallocated_output = std::nullopt);
// Perhaps can add overloads for shard spec arguments later

}  // namespace ttnn

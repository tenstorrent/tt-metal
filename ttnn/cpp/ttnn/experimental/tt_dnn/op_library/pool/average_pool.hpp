// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tt_metal/host_api.hpp"
#include "tensor/tensor.hpp"

#include "ttnn/operation.hpp"

namespace tt {
namespace tt_metal {

enum class PoolType {
    AVG
};

Tensor average_pool_2d(const Tensor& input, const MemoryConfig& output_mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG, const std::optional<DataType>& output_dtype = std::nullopt);

}  // namespace tt_metal
}  // namespace tt

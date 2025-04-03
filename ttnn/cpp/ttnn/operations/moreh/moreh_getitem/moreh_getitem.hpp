// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>

#include <tt-metalium/small_vector.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/operations/moreh/moreh_getitem/device/moreh_getitem_device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

namespace ttnn::operations::moreh::moreh_getitem {
struct MorehGetItem {
    static Tensor invoke(
        const std::optional<const Tensor>& input,
        const std::vector<Tensor>& index_tensors,
        const ttnn::SmallVector<uint32_t>& index_dims,
        const std::optional<Tensor>& output,
        // const CoreRange core_range,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_getitem

namespace ttnn {
constexpr auto moreh_getitem = ttnn::register_operation_with_auto_launch_op<
    "ttnn::moreh_getitem",
    ttnn::operations::moreh::moreh_getitem::MorehGetItem>();
}

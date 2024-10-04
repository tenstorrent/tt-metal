// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "full_like.hpp"
#include <variant>
#include "ttnn/operations/full_like/device/full_like_device_operation.hpp"

namespace ttnn::operations::full_like {


    Tensor FullLike::invoke(
        const Tensor &input,
        const int fill_value,
        const std::optional<DataType> &dtype,
        const std::optional<Layout> &layout,
        const std::optional<MemoryConfig> &memory_config) {
            return ttnn::prim::full_like_2(
                input, fill_value, dtype, layout, memory_config);
        }

}

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange.hpp"

#include "device/moreh_arange_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_arange {
Tensor MorehArange::invoke(float start,
                           float end,
                           float step,
                           const Tensor& any,
                           const std::optional<Tensor>& output,
                           bool untilize_out,
                           const std::optional<DataType>& dtype,
                           const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::moreh_arange(start, end, step, any, output, untilize_out, dtype, memory_config);
}
}  // namespace ttnn::operations::moreh::moreh_arange

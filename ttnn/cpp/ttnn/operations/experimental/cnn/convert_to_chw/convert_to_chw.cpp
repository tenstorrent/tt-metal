// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_op.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn::operations::experimental::cnn {

ttnn::Tensor ExecuteConvertToCHW::invoke(
    QueueId queue_id,
    const Tensor& a,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType>& dtype) {
    auto program = ConvertToCHW{memory_config.value_or(a.memory_config()), dtype.value_or(a.dtype())};
    return tt::tt_metal::operation::run(program, {a}, {}, {}, queue_id).at(0);
}

}  // namespace ttnn::operations::experimental::cnn

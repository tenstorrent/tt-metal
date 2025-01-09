// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_op.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::experimental::cnn {

ttnn::Tensor ExecuteConvertToCHW::invoke(
    uint8_t queue_id,
    const Tensor& a,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType>& dtype) {
    auto program = ConvertToCHW{memory_config.value_or(a.memory_config()), dtype.value_or(a.dtype())};
    return operation::run(program, {a}, {}, {}, queue_id).at(0);
}

ttnn::Tensor ExecuteConvertToCHW::invoke(
    const Tensor& a, const std::optional<MemoryConfig>& memory_config, const std::optional<DataType>& dtype) {
    return invoke(DefaultQueueId, a, memory_config, dtype);
}

}  // namespace ttnn::operations::experimental::cnn

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_op.hpp"

namespace ttnn::operations::experimental::cnn {

ttnn::Tensor ExecuteConvertToCHW::invoke(const Tensor& a, const std::optional<DataType>& dtype) {
    auto output_memory_config = infer_output_memory_config(a);

    auto program = ConvertToCHW{output_memory_config, dtype.value_or(a.dtype())};
    return tt::tt_metal::operation::run(program, {a}, {}, {}).at(0);
}

}  // namespace ttnn::operations::experimental::cnn

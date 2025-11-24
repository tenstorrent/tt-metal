// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_device_operation.hpp"

namespace ttnn::operations::experimental::cnn {

ttnn::Tensor ExecuteConvertToCHW::invoke(const Tensor& a, const std::optional<DataType>& dtype) {
    return ttnn::prim::convert_to_chw(a, dtype);
}

}  // namespace ttnn::operations::experimental::cnn

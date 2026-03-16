// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor convert_to_chw(const Tensor& input, const std::optional<DataType>& dtype) {
    return ttnn::prim::convert_to_chw(input, dtype);
}

}  // namespace ttnn::experimental

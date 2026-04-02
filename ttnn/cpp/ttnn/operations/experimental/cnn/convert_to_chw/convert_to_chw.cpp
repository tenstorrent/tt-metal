// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "convert_to_chw.hpp"

#include "device/convert_to_chw_device_operation.hpp"
#include "ttnn/graph/composite_trace.hpp"

namespace ttnn::experimental {

ttnn::Tensor convert_to_chw(const Tensor& input, const std::optional<DataType>& dtype) {
    ttnn::graph::ScopedCompositeTrace _trace("ttnn::experimental::convert_to_chw");
    return ttnn::prim::convert_to_chw(input, dtype);
}

}  // namespace ttnn::experimental

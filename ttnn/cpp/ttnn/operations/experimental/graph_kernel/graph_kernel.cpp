// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/graph_kernel/graph_kernel.hpp"

#include "device/graph_kernel_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor graph_kernel(const std::vector<Tensor>& inputs, const std::string& text) {
    return ttnn::prim::graph_kernel(inputs, text);
}

}  // namespace ttnn::operations::experimental

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/hello_world/hello_world.hpp"

#include "device/hello_world_device_operation.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor hello_world(const Tensor& input_tensor) { return ttnn::prim::hello_world(input_tensor); }

}  // namespace ttnn::operations::experimental

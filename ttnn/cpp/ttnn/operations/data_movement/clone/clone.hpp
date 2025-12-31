// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/clone_device_operation.hpp"

namespace ttnn {
// Expose prim::clone directly as ttnn::clone
// The prim function is the actual implementation with tracing via device_operation::launch
using prim::clone;
}  // namespace ttnn

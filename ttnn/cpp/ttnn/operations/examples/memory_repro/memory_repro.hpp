// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device/memory_repro_device_operation.hpp"

namespace ttnn {

Tensor memory_repro(const Tensor& input_tensor);

}  // namespace ttnn

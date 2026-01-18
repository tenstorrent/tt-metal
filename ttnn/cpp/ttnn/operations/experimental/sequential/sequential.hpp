// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Sequential operation for executing multiple operations in order.
//
// Current implementation: Python-based sequential execution.
// Future: C++ fused execution with CB chaining for efficiency.
//
// Python usage:
//   results = ttnn.sequential([
//       (ttnn.rms_norm, input1, {"epsilon": 1e-5}),
//       (ttnn.layer_norm, input2),
//   ])

#include "device/sequential_device_operation.hpp"

namespace ttnn::operations::experimental::sequential {

// Reserved for future C++ sequential execution API.

}  // namespace ttnn::operations::experimental::sequential

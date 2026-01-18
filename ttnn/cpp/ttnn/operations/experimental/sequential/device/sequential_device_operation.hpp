// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Minimal device operation for sequential execution.
// This is a placeholder for future CB chaining optimizations.
// Currently, sequential execution is handled entirely in Python,
// calling each operation in order without program fusion.

#include "sequential_device_operation_types.hpp"

namespace ttnn::operations::experimental::sequential {

// Future: SequentialDeviceOperation that fuses multiple operations
// into a single program with shared circular buffers.
//
// For now, this namespace is reserved for future development.

}  // namespace ttnn::operations::experimental::sequential

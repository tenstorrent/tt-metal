// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Combined convenience header — includes both reader and writer helpers.
//
// IMPORTANT: Do NOT include this header in a writer-only kernel.
// The reader helper (read_matmul_tiles) uses chained TensorAccessorArgs for two
// tensors. If the writer kernel only provides compile-time args for one tensor
// accessor, the reader template will fail to compile when instantiated.
//
// Instead, include only what you need:
//   Reader kernels:  #include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_reader_helpers.hpp"
//   Writer kernels:  #include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_writer_helpers.hpp"
//
// This combined header is provided for kernels that use both helpers (e.g., a
// single-kernel reader+writer), or as a convenience when the compile-time arg
// layout provides both accessor arg blocks.

#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_reader_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/matmul_1d_writer_helpers.hpp"

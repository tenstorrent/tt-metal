// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

// LayerNorm - Reader Kernel
//
// Three-pass pattern:
//   Startup : Fill c_1 (scaler 1/W) and c_2 (epsilon) once.
//   Pass 1  : Stream Wt input tiles -> c_0 (compute uses for mean).
//   Pass 2  : Re-stream Wt input tiles -> c_0 (compute uses for variance).
//   Pass 3  : Re-stream Wt input tiles -> c_0, plus gamma -> c_3, beta -> c_4.
//
// Compile-time args (via TensorAccessorArgs):
//   [0+] : TensorAccessorArgs for input tensor
//   [N+] : TensorAccessorArgs for gamma tensor (if has_gamma)
//   [M+] : TensorAccessorArgs for beta tensor  (if has_beta)
//
// Runtime args:
//   [0] input_addr        : uint32  Input buffer base address
//   [1] num_rows_per_core : uint32  Number of tile-rows for this core
//   [2] Wt               : uint32  Width in tiles
//   [3] tile_offset      : uint32  Starting tile index in the input buffer
//   [4] gamma_addr       : uint32  Gamma buffer address (0 if no gamma)
//   [5] beta_addr        : uint32  Beta buffer address  (0 if no beta)
//   [6] eps_bits         : uint32  Epsilon packed as (bf16 << 16 | bf16)
//   [7] scaler_bits      : uint32  1/W   packed as (bf16 << 16 | bf16)

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // TODO: Implement reader kernel
    // Stub: do nothing; compute and writer stubs will also do nothing,
    //       so the output will contain whatever was pre-allocated.
}

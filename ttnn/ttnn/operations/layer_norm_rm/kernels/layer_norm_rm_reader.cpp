// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// layer_norm_rm — Reader Kernel (STUB)
//
// Stage 1 (data_pipeline): reads RM input sticks and packs into tile-sized CB pages.
// Stage 4 (affine_transform): additionally reads gamma/beta RM sticks.
//
// Compile-time args:
//   [0]  stick_size          — W * sizeof(bfloat16) bytes per RM stick
//   [1+] TensorAccessorArgs(input) — interleaved input accessor
//
// Runtime args:
//   [0] src_addr       — input buffer address
//   [1] start_stick_id — first RM stick for this core
//   [2] num_sticks     — nblocks * 32 sticks to read
//   [3] gamma_addr     — gamma buffer address (0 if has_gamma==0)
//   [4] beta_addr      — beta buffer address (0 if has_beta==0)

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    // Stub: no-op. The real implementation will:
    // 1. Fill cb_scaler once with prepare_reduce_scaler<cb_scaler>(1.0f / W)
    // 2. Loop over nblocks: for each block, pack 32 RM sticks into Wt tile-sized
    //    CB pages in cb_in_rm using noc_async_read per stick.
    // 3. (Stage 4) Fill cb_gamma and cb_beta from gamma/beta tensors.
}

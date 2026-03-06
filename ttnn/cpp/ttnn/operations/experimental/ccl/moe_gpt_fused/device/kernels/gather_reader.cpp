// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Gather Reader (RISCV_1 / NOC_1) for moe_gpt_fused
//
// Input tensor is WIDTH_SHARDED across 3 tilize cores.
// CB c_7 is backed by the input tensor shard, so data is already in L1.
// Just push the CB to signal compute that input is ready.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "moe_gpt_fused_ring_common.h"

void kernel_main() {
    constexpr auto tilize_input_cb = tt::CBIndex::c_7;
    constexpr uint32_t tokens_per_chunk = moe_gpt_fused_ring::TOKENS_PER_CHUNK;  // 32

    // Data is already in L1 (WIDTH_SHARDED input tensor backs c_7).
    // Push to signal compute kernel that input is available.
    cb_push_back(tilize_input_cb, tokens_per_chunk);
}

// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// Writer for scaled_dot_product_attention (Flash Attention).
//
// Stage 0 (init): No output to write yet. The writer is a stub — it returns
// immediately. Later stages drain cb_o to the DRAM output buffer after all
// KV-blocks for a Q-block are processed.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // Stage 0: no-op. The output drain (cb_o → DRAM) is added in phase 14.
}

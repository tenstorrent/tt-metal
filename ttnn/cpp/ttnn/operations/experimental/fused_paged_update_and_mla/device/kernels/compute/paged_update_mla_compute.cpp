// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "compute_kernel_api/common.h"

namespace NAMESPACE {
void MAIN {
    uint32_t kv_cb_index = tt::CB::c_in1;
    
    for (uint32_t i = 0; i < 32; i++) {
        cb_wait_front(kv_cb_index, 1);
        cb_pop_front(kv_cb_index, 1);
    }
}
}

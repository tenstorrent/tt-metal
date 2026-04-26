// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

// Minimal BRISC reader for the sharded decode RoPE path.
// The compute kernel (rotary_embedding_hf_sharded.cpp) relies on a scalar CB that
// contains -1.0 in bfloat16 for the rotate-half negation.  No NOC transfers are
// required (all tensor data lives in globally-allocated L1 CBs), so this kernel's
// only job is to write the scalar value into L1 and signal the CB as ready.

void kernel_main() {
    constexpr uint32_t scalar_cb_id = get_compile_time_arg_val(0);
    // bfloat16 representation of -1.0f, passed as uint32 from the factory.
    constexpr uint16_t scalar_value = (uint16_t)get_compile_time_arg_val(1);

    cb_reserve_back(scalar_cb_id, 1);
    uint32_t l1_write_addr = get_write_ptr(scalar_cb_id);
    volatile tt_l1_ptr uint16_t* scalar_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    scalar_buf[0] = scalar_value;
    cb_push_back(scalar_cb_id, 1);
}

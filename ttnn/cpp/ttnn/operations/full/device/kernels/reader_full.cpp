// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

typedef union {
    float f;
    uint32_t u;
} u;

void kernel_main() {
    uint32_t fill_value = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_value = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;
    u val;
    val.u = fill_value;

    cb_reserve_back(cb_value, onetile);

    uint32_t write_addr = get_write_ptr(cb_value);

#ifdef OUTPUT_DTYPE_BFLOAT16
    auto ptr = reinterpret_cast<uint16_t *>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = val.u >> 16;
    }
#endif
#ifdef OUTPUT_DTYPE_INT32
    auto ptr = reinterpret_cast<uint32_t *>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = fill_value;
    }
#endif
#ifdef OUTPUT_DTYPE_FLOAT32
    auto ptr = reinterpret_cast<float *>(write_addr);
    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = val.f;
    }
#endif
    cb_push_back(cb_value, 1);
}

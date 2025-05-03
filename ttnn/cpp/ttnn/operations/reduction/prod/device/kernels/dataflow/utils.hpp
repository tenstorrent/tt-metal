// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "dataflow_api.h"

void fill_cb_with_value(uint32_t cb_id, uint32_t value, int32_t num_of_elems = 1024) {
    cb_reserve_back(cb_id, 1);
    volatile tt_l1_ptr std::uint16_t* ptr = (volatile tt_l1_ptr uint16_t*)(get_write_ptr(cb_id));
    for (int j = 0; j < num_of_elems; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
    cb_push_back(cb_id, 1);
}

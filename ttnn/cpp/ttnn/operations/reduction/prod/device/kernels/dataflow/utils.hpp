// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"

void fill_cb_with_value(uint32_t dfb_id, uint32_t value, int32_t num_of_elems = 1024) {
    DataflowBuffer dfb(dfb_id);
    dfb.reserve_back(1);
    CoreLocalMem<volatile std::uint16_t> ptr(dfb.get_write_ptr());
    for (int j = 0; j < num_of_elems; j++) {
        ptr[j] = uint16_t(value >> 16);
    }
    dfb.push_back(1);
}

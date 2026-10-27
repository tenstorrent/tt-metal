// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// reader_unary_sharded_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and this file is
// retired, changes here likely belong in the fork too.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"

#include "api/debug/dprint.h"

void kernel_main() {
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);

    DataflowBuffer dfb(cb_id_in0);
    dfb.push_back(num_tiles_per_core);
}

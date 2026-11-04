// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert, the schedule leader: builds the call's schedule from the DRAM rows (se_dyn_load); with the
// schedule pull (SE_DYN_PULL) its compact copy of this chip's counts / regions is what every other core pulls.
//
// CT: 0 NUM_E
// RT: the se_dyn.hpp args
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_dyn.hpp"

void kernel_main() {
    constexpr uint32_t num_e = get_compile_time_arg_val(0);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_7);
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, 0, scratch, 1);
    noc_async_full_barrier();
}

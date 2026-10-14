// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// Flat spatial expert, DRISC mode (se_drisc.hpp): the launcher core posts this call's gate/up jobs to the DRISCs
// (se_drisc_post.hpp) from the call's schedule (se_dyn_load, as the readers build it).
//
// CT: 0 NUM_E
// RT: the se_drisc_post.hpp args from 0, then the se_dyn.hpp args
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "se_drisc_post.hpp"

void kernel_main() {
    constexpr uint32_t num_e = get_compile_time_arg_val(0);
    const uint32_t scratch = get_write_ptr(tt::CBIndex::c_7);
    SeDyn dyn;
    se_dyn_load<num_e>(dyn, se_drisc_post_nargs(0), scratch, 1);
    se_drisc_post(dyn, 0, scratch);
    noc_async_full_barrier();
}

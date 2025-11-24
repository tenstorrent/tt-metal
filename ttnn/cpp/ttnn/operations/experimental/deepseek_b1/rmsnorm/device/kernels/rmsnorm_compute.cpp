// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "rmsnorm_compute_utils.hpp"

namespace NAMESPACE {
void MAIN {
    uint32_t arg_idx = 0;
    DEFINE_RMSNORM_COMPUTE_VARS(rms);

    // Init block done only once
    binary_op_init_common(rms_input_cb, rms_input_cb, rms_output_cb);
    cb_wait_front(rms_scalars_cb, 2);
    cb_wait_front(rms_gamma_cb, rms_num_tiles);  // we don't pop, only wait once and reuse
    rsqrt_tile_init();                           // this is the only sfpu op we use, so we init once

    COMPUTE_RMSNORM(rms, true);
}
}  // namespace NAMESPACE

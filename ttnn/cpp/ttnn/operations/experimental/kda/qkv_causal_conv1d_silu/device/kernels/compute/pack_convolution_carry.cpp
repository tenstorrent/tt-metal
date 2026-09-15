// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "api/compute/compute_kernel_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"

TT_KERNEL void compute(uint32_t ct_count) {
    compute_kernel_hw_startup(dfb::packed_rm, dfb::packed_tile);
    for (uint32_t ct = 0; ct < ct_count; ++ct) {
        compute_kernel_lib::tilize<1, dfb::packed_rm, dfb::packed_tile>(1);
    }
}

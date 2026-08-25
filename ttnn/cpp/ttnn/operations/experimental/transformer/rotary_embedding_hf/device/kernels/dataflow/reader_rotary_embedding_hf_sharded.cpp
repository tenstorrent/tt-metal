// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Minimal BRISC reader for the sharded decode RoPE path.
// The compute kernel (rotary_embedding_hf_sharded.cpp) relies on a scalar DFB that
// contains -1.0 in bfloat16 for the rotate-half negation.  No NOC transfers are
// required (all tensor data lives in globally-allocated L1 buffers), so this kernel's
// only job is to write the scalar value into L1 and signal the DFB as ready.

void kernel_main() {
    // bfloat16 representation of -1.0f, passed as uint32 from the factory.
    constexpr uint16_t scalar_value = (uint16_t)get_arg(args::scalar_value);

    DataflowBuffer dfb_scalar(dfb::scalar);

    dfb_scalar.reserve_back(1);
    uint32_t l1_write_addr = dfb_scalar.get_write_ptr();
    volatile tt_l1_ptr uint16_t* scalar_buf = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
    scalar_buf[0] = scalar_value;
    dfb_scalar.push_back(1);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "mhc_sinkhorn_permute.h"

// Sharded-input reader: mixes are already resident in this core's L1 (the mixes CB aliases
// the shard), so we only fetch the 8 constant tiles from DRAM and signal the mixes as ready.
// With SFPU_SINKHORN (compile-time arg 2) it then permutes each token-tile's comb logits into
// the SFPU layout and the Sinkhorn result back into token rows (mhc_sinkhorn_permute.h).
void kernel_main() {
    const uint32_t consts_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_token_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_mixes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_consts = get_compile_time_arg_val(1);
    constexpr bool sfpu_sinkhorn = get_compile_time_arg_val(2) != 0;
    constexpr auto consts_args = TensorAccessorArgs<3>();
    constexpr uint32_t num_const_tiles = 8;

    const uint32_t consts_page = get_local_cb_interface(cb_consts).fifo_page_size;
    const auto s_consts = TensorAccessor(consts_args, consts_addr);

    Noc noc;
    CircularBuffer cb_c(cb_consts);
    CircularBuffer cb_m(cb_mixes);

    cb_c.reserve_back(num_const_tiles);
    for (uint32_t i = 0; i < num_const_tiles; ++i) {
        noc.async_read(s_consts, cb_c, consts_page, {.page_id = i}, {.offset_bytes = i * consts_page});
    }
    noc.async_read_barrier();
    cb_c.push_back(num_const_tiles);

    cb_m.push_back(num_token_tiles);  // mixes already in L1 via the aliased CB

    if constexpr (sfpu_sinkhorn) {
        for (uint32_t t = 0; t < num_token_tiles; ++t) {
            mhc_sinkhorn_permute_in(noc);
            mhc_sinkhorn_permute_out(noc);
        }
    }
}

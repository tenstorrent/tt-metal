// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "in1_matmul.hpp"

void kernel_main() {
    Noc noc;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    constexpr uint32_t tile_bytes_bf16 = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t tile_bytes_bfp4 = CT_ARG("tile_bytes_bfp4");
    Chain chain = CHAIN("in1");
    CircularBuffer cb_y(down.cb_out);

    const auto y = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_y")>(), RT_ARG("y_addr"), tile_bytes_bf16);

    chain.valid.set(VALID);

    bool write_pending = false;
    bool write_in_flight = false;
    uint32_t pending_m = 0;
    const uint32_t defer_k_block = RT_ARG("defer_k_block");
    for (uint32_t m_block = 0; m_block < CT_ARG("M_blocks"); m_block++) {
        const uint32_t m_tile = RT_ARG("M_start") + m_block * gate_up.M_block;
        bool k_forward = true;
        for (uint32_t n_block = 0; n_block < gate_up.N_blocks; n_block++) {
            for (uint32_t k_iter = 0; k_iter < gate_up.K_blocks; k_iter++) {
                if (write_in_flight) {
                    retire_out_block(noc, down, cb_y);
                    write_in_flight = false;
                }
                if (write_pending && k_iter == defer_k_block) {
                    issue_out_block_writes(noc, down, y, tile_bytes_bf16, cb_y, pending_m, RT_ARG("mm2_N_start"));
                    write_pending = false;
                    write_in_flight = true;
                }
                stream_in1_block(
                    noc,
                    gate_up,
                    chain,
                    RT_ARG("w_gate_up_addr"),
                    tile_bytes_bfp4,
                    k_forward ? k_iter : gate_up.K_blocks - 1 - k_iter,
                    RT_ARG("mm1_N_start") + n_block * gate_up.N_block);
            }
            k_forward = !k_forward;
        }
        for (uint32_t k_block = 0; k_block < down.K_blocks; k_block++) {
            stream_in1_block(noc, down, chain, RT_ARG("w_down_addr"), tile_bytes_bfp4, k_block, RT_ARG("mm2_N_start"));
        }
        write_pending = true;
        pending_m = m_tile;
    }
    if (write_in_flight) {
        retire_out_block(noc, down, cb_y);
    }
    issue_out_block_writes(noc, down, y, tile_bytes_bf16, cb_y, pending_m, RT_ARG("mm2_N_start"));
    retire_out_block(noc, down, cb_y);
    noc.async_write_barrier();
}

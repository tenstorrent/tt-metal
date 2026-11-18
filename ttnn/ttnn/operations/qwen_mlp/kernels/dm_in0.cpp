// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "group_barrier.hpp"
#include "in0_matmul.hpp"
#include "rmsnorm_dm.hpp"

void kernel_main() {
    Noc noc;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    constexpr uint32_t tile_bytes = CT_ARG("tile_bytes_bf16");
    Chain chain = CHAIN("in0");
    Semaphore<> norm_written(CT_ARG("sem_barrier_norm"));
    Semaphore<> gate_up_written(CT_ARG("sem_barrier_gate_up"));

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes);
    const auto w_norm = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes);
    const auto n = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_n")>(), RT_ARG("n_addr"), tile_bytes);
    const auto a = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_a")>(), RT_ARG("a_addr"), tile_bytes);

    chain.valid.set(VALID);

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes);
    read_norm_rows(
        noc,
        h,
        w_norm,
        CT_ARG("cb_norm_in"),
        CT_ARG("cb_norm_weight"),
        gate_up.in0_row_tiles,
        CT_ARG("norm_chunk"),
        tile_bytes,
        RT_ARG("norm_first_row"),
        CT_ARG("norm_row_stride"),
        RT_ARG("norm_rows"));

    if (chain.is_injector) {
        wait_group_written(norm_written);
    }
    in0_matmul_phase(noc, gate_up, MATMUL_RANGE("mm1"), chain, n, tile_bytes);

    if (chain.is_injector) {
        wait_group_written(gate_up_written);
    }
    in0_matmul_phase(noc, down, MATMUL_RANGE("mm2"), chain, a, tile_bytes);
}

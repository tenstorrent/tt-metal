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
    Semaphore<> norm_done(CT_ARG("sem_barrier_norm"));
    Semaphore<> gate_up_done(CT_ARG("sem_barrier_gate_up"));

    const auto n = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_n")>(), RT_ARG("n_addr"), tile_bytes);
    const auto a = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_a")>(), RT_ARG("a_addr"), tile_bytes);
    const auto y = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_y")>(), RT_ARG("y_addr"), tile_bytes);

    chain.valid.set(VALID);

    write_norm_rows(
        noc,
        n,
        CT_ARG("cb_norm_out"),
        gate_up.in0_row_tiles,
        CT_ARG("norm_group"),
        tile_bytes,
        RT_ARG("norm_first_row"),
        CT_ARG("norm_row_stride"),
        RT_ARG("norm_rows"));
    group_barrier(noc, norm_done, chain.is_injector);

    in0_matmul_phase(noc, gate_up, MATMUL_RANGE("mm1"), chain, n, a, tile_bytes, CT_ARG("defer_k_block"));
    group_barrier(noc, gate_up_done, chain.is_injector);

    in0_matmul_phase(noc, down, MATMUL_RANGE("mm2"), chain, a, y, tile_bytes, CT_ARG("defer_k_block"));
    noc.async_atomic_barrier();
}

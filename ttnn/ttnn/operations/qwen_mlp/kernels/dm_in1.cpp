// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "group_barrier.hpp"
#include "in1_matmul.hpp"
#include "rmsnorm_dm.hpp"

void kernel_main() {
    Noc noc;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    constexpr uint32_t tile_bytes_bf16 = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t tile_bytes_bfp4 = CT_ARG("tile_bytes_bfp4");
    Chain chain = CHAIN("in1");
    Semaphore<> norm_written(CT_ARG("sem_barrier_norm"));
    Semaphore<> gate_up_written(CT_ARG("sem_barrier_gate_up"));

    const auto n = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_n")>(), RT_ARG("n_addr"), tile_bytes_bf16);
    const auto a = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_a")>(), RT_ARG("a_addr"), tile_bytes_bf16);
    const auto y = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_y")>(), RT_ARG("y_addr"), tile_bytes_bf16);

    chain.valid.set(VALID);

    write_norm_rows(
        noc,
        n,
        CT_ARG("cb_norm_out"),
        gate_up.in0_row_tiles,
        CT_ARG("norm_group"),
        tile_bytes_bf16,
        RT_ARG("norm_first_row"),
        CT_ARG("norm_row_stride"),
        RT_ARG("norm_rows"));
    signal_group_written(noc, norm_written);

    in1_matmul_phase(
        noc,
        gate_up,
        MATMUL_RANGE("mm1"),
        chain,
        RT_ARG("w_gate_up_addr"),
        a,
        tile_bytes_bfp4,
        tile_bytes_bf16,
        RT_ARG("defer_k_block"));
    signal_group_written(noc, gate_up_written);

    in1_matmul_phase(
        noc,
        down,
        MATMUL_RANGE("mm2"),
        chain,
        RT_ARG("w_down_addr"),
        y,
        tile_bytes_bfp4,
        tile_bytes_bf16,
        RT_ARG("defer_k_block"));
}

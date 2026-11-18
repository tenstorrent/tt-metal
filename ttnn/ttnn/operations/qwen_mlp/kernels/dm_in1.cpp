// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "in1_matmul.hpp"
#include "rmsnorm_dm.hpp"

void kernel_main() {
    Noc noc;
    constexpr MatmulShape gate_up = MATMUL_SHAPE("mm1");
    constexpr MatmulShape down = MATMUL_SHAPE("mm2");
    constexpr uint32_t tile_bytes_bf16 = CT_ARG("tile_bytes_bf16");
    constexpr uint32_t tile_bytes_bfp4 = CT_ARG("tile_bytes_bfp4");
    Chain chain = CHAIN("in1");

    const auto h = TensorAccessor(TensorAccessorArgs<CT_ARG("ta_h")>(), RT_ARG("h_addr"), tile_bytes_bf16);
    const auto w_norm =
        TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_norm")>(), RT_ARG("w_norm_addr"), tile_bytes_bf16);
    const auto w_gate_up =
        TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_gate_up")>(), RT_ARG("w_gate_up_addr"), tile_bytes_bfp4);
    const auto w_down =
        TensorAccessor(TensorAccessorArgs<CT_ARG("ta_w_down")>(), RT_ARG("w_down_addr"), tile_bytes_bfp4);

    chain.valid.set(VALID);

    fill_reduce_scaler(noc, CT_ARG("cb_norm_scaler"), tile_bytes_bf16);
    read_norm_rows(
        noc,
        h,
        w_norm,
        CT_ARG("cb_norm_in"),
        CT_ARG("cb_norm_weight"),
        gate_up.in0_row_tiles,
        CT_ARG("norm_chunk"),
        tile_bytes_bf16,
        RT_ARG("norm_first_row"),
        CT_ARG("norm_row_stride"),
        RT_ARG("norm_rows"));

    in1_matmul_phase(noc, gate_up, MATMUL_RANGE("mm1"), chain, w_gate_up, tile_bytes_bfp4);
    in1_matmul_phase(noc, down, MATMUL_RANGE("mm2"), chain, w_down, tile_bytes_bfp4);
    noc.async_write_barrier();
}

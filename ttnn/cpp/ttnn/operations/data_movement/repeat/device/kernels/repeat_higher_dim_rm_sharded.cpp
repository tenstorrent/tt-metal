// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RM higher-dim sharded; noc_async_*_sharded splits cross-shard rows.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using namespace tt::data_movement::common;

void kernel_main() {
    const auto higher_dim_start = get_arg(args::higher_dim_start);
    const auto higher_dim_end = get_arg(args::higher_dim_end);
    const auto lower_dim_start = get_arg(args::lower_dim_start);
    const auto lower_dim_end = get_arg(args::lower_dim_end);
    const auto repetitions = get_arg(args::repetitions);
    const auto nop = get_arg(args::nop);

    constexpr auto original_page_size_bytes = get_arg(args::original_page_size_bytes);
    constexpr auto LOWER_DIMS = get_arg(args::LOWER_DIMS);
    constexpr auto REP_DIM = get_arg(args::REP_DIM);

    constexpr uint32_t LOWER_DIMS_TIMES_REP_DIM = LOWER_DIMS * REP_DIM;

    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(tensor::src);
    const auto d = TensorAccessor(tensor::dst);

    DataflowBuffer dfb(dfb::in0);
    dfb.reserve_back(1);
    const uint32_t cb_slot = dfb.get_write_ptr();
    dfb.push_back(1);

    Noc noc;

    for (uint32_t h = higher_dim_start; h < higher_dim_end; h++) {
        const uint32_t h_offset = h * LOWER_DIMS_TIMES_REP_DIM;
        const uint32_t h_offset_rep = h_offset * repetitions;
        for (uint32_t r = 0; r < REP_DIM; r++) {
            const uint32_t r_offset = r * LOWER_DIMS;
            for (uint32_t l = lower_dim_start; l < lower_dim_end; l++) {
                const uint32_t read_offset = h_offset + r_offset + l;
                noc_async_read_sharded(noc, cb_slot, s, read_offset, 0, original_page_size_bytes);
                noc.async_read_barrier();
                for (uint32_t n = 0; n < repetitions; n++) {
                    const uint32_t write_offset = h_offset_rep + n * LOWER_DIMS_TIMES_REP_DIM + r_offset + l;
                    noc_async_write_sharded(noc, cb_slot, d, write_offset, 0, original_page_size_bytes);
                }
                noc.async_write_barrier();
            }
        }
    }
}

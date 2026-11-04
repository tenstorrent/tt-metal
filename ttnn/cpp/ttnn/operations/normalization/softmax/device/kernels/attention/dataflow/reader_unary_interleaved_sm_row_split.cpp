// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    const std::uint32_t T = get_arg(args::T);
    const std::uint32_t t0 = get_arg(args::t0);
    const std::uint32_t num_rows = get_arg(args::num_rows);
    const std::uint32_t Wt = get_arg(args::Wt);
    const std::uint32_t blk = get_arg(args::blk);
    const std::uint32_t pad = get_arg(args::pad);
    const std::uint32_t row0 = get_arg(args::row0);
    const std::uint32_t rstride = get_arg(args::rstride);
    if (T == 0) {
        return;
    }

    DataflowBuffer in0(dfb::in0);
    const std::uint32_t tile_bytes = in0.get_entry_size();
    const auto src = TensorAccessor(tensor::src);

    DataflowBuffer max_scaler(dfb::max_scaler);
    DataflowBuffer sum_scaler(dfb::sum_scaler);
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>(
            max_scaler);
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            sum_scaler);

    const Noc noc;
    for (std::uint32_t r = 0; r < num_rows; ++r) {
        std::uint32_t tile = (row0 + r * rstride) * Wt + t0;
#ifdef READ_BATCH
        const std::uint32_t rbk = READ_BATCH;
#else
        const std::uint32_t rbk = blk;
#endif
        for (std::uint32_t j = 0; j < T; j += rbk) {
            const std::uint32_t rem = (j + rbk > T) ? (T - j) : rbk;
            in0.reserve_back(static_cast<uint16_t>(rem));
            std::uint32_t off = 0;
            for (std::uint32_t i = 0; i < rem; ++i) {
                noc.async_read(src, in0, tile_bytes, {.page_id = tile}, {.offset_bytes = off});
                ++tile;
                off += tile_bytes;
            }
            noc.async_read_barrier();
            in0.push_back(static_cast<uint16_t>(rem));
        }
        if (pad > 0) {
            in0.reserve_back(static_cast<uint16_t>(pad));
            in0.push_back(static_cast<uint16_t>(pad));
        }
    }
}

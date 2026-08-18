// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto K = get_arg(args::K);
    constexpr uint32_t Kt = K % 32 == 0 ? K / 32 : K / 32 + 1;

    // can amortize the noc reads by doing them side by side for the two tensors
    constexpr uint32_t onetile = 1;

    // Reduce ops need to multiply by a scalar. We always want to multiply by 1.0f
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<dfb::scale, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    const auto interleaved_accessor0 = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_out(dfb::out);

    const uint32_t tile_bytes = dfb_out.get_tile_size();

    uint32_t tile_id = 0;
    dfb_out.wait_front(Ht * Kt);
    for (uint32_t j = 0; j < Ht; ++j) {
        for (uint32_t i = 0; i < Kt; ++i) {
            noc.async_write(
                dfb_out,
                interleaved_accessor0,
                tile_bytes,
                {.offset_bytes = tile_id * tile_bytes},
                {.page_id = tile_id});
            tile_id++;
        }
    }
    noc.async_write_barrier();
    dfb_out.pop_front(Ht * Kt);
}

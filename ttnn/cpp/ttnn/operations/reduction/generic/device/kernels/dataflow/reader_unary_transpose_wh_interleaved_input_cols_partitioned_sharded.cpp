// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t batch = get_arg(args::batch);
    uint32_t row_size_bytes = get_arg(args::row_size_bytes);
    uint32_t batch_size_bytes = get_arg(args::batch_size_bytes);

#ifdef REDUCE_SCALER
    using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();
#endif

    // The host fixes the stream order to match the planned compute call.
    constexpr uint32_t row_chunk = get_arg(args::reduce_output_tiles);

    constexpr uint32_t onetile = 1;

    Noc noc;
    // dfb::in0 is the reduce input pipe: this kernel fills it, the compute kernel drains it.
    DataflowBuffer dfb_in0(dfb::in0);
    // dfb::in1 is a view onto the resident input shard (borrowed memory). This kernel is its only
    // toucher: it reserves the whole shard and then re-reads it as the NoC source below.
    DataflowBuffer dfb_in1(dfb::in1);
    uint32_t tile_bytes = dfb_in0.get_tile_size();

    dfb_in1.reserve_back(num_tiles);
    uint32_t base_l1_addr = dfb_in1.get_write_ptr();

    UnicastEndpoint src;
    uint32_t src_noc_x = my_x[noc_index];
    uint32_t src_noc_y = my_y[noc_index];

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < Wt; i += row_chunk) {
            uint32_t chunk_end = (i + row_chunk < Wt) ? (i + row_chunk) : Wt;
            for (uint32_t j = 0; j < Ht; ++j) {
                uint32_t row_l1_addr = base_l1_addr + j * row_size_bytes;
                for (uint32_t k = i; k < chunk_end; ++k) {
                    dfb_in0.reserve_back(onetile);
                    noc.async_read(
                        src,
                        dfb_in0,
                        tile_bytes,
                        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = row_l1_addr + k * tile_bytes},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    dfb_in0.push_back(onetile);
                }
            }
        }
        base_l1_addr += batch_size_bytes;
    }
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reduction-op width-sharded reader, ported to Metal 2.0.
//
// Host bindings expected (per the H factory's width-sharded KernelSpec):
//   compile_time_arg_bindings: { {"scaler_bits", ...} }
//   runtime_arguments_schema.named_runtime_args:
//     { "num_tiles", "Wt", "Ht", "batch", "row_size_bytes", "batch_size_bytes" }
//   dfb_bindings:
//     { INPUT (CONSUMER, name="input"),
//       INPUT_SHARDED (PRODUCER, name="input_sharded"),  // borrowed-memory DFB
//       SCALER (PRODUCER, name="scaler") }
//   tensor_bindings: (none - inputs read from the borrowed-memory DFB only)
//
// REDUCE_SCALER is always defined for this kernel today (the H width-sharded
// factory always sets the define). The legacy kernel kept the ifdef in case
// the kernel was reused elsewhere; we keep the gate to match.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    auto num_tiles = get_arg(args::num_tiles);
    auto Wt = get_arg(args::Wt);
    auto Ht = get_arg(args::Ht);
    auto batch = get_arg(args::batch);
    auto row_size_bytes = get_arg(args::row_size_bytes);
    auto batch_size_bytes = get_arg(args::batch_size_bytes);

#ifdef REDUCE_SCALER
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);
#endif

    // Emit tiles in N, W_skip, H, W_chunk order to match the chunked iteration of the
    // unified reduce compute kernel (row_chunk = DEST_AUTO_LIMIT). For shard_Wt=1 this
    // degenerates to one column per chunk; for shard_Wt>1 it interleaves columns.
    constexpr uint32_t row_chunk = compute_kernel_lib::DEST_AUTO_LIMIT;

    constexpr uint32_t onetile = 1;

    DataflowBuffer cb_in0(dfb::input);
    DataflowBuffer cb_in1(dfb::input_sharded);

    uint32_t tile_bytes = cb_in0.get_tile_size();

    Noc noc;

    cb_in1.reserve_back(num_tiles);
    uint32_t base_l1_addr = cb_in1.get_write_ptr();

    UnicastEndpoint src;
    uint32_t src_noc_x = my_x[noc_index];
    uint32_t src_noc_y = my_y[noc_index];

    for (uint32_t b = 0; b < batch; ++b) {
        for (uint32_t i = 0; i < Wt; i += row_chunk) {
            uint32_t chunk_end = (i + row_chunk < Wt) ? (i + row_chunk) : Wt;
            for (uint32_t j = 0; j < Ht; ++j) {
                uint32_t row_l1_addr = base_l1_addr + j * row_size_bytes;
                for (uint32_t k = i; k < chunk_end; ++k) {
                    cb_in0.reserve_back(onetile);
                    noc.async_read(
                        src,
                        cb_in0,
                        tile_bytes,
                        {.noc_x = src_noc_x, .noc_y = src_noc_y, .addr = row_l1_addr + k * tile_bytes},
                        {.offset_bytes = 0});
                    noc.async_read_barrier();
                    cb_in0.push_back(onetile);
                }
            }
        }
        base_l1_addr += batch_size_bytes;
    }
}

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
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t Wt = get_arg(args::Wt);
    uint32_t Ht = get_arg(args::Ht);
    uint32_t batch = get_arg(args::batch);
    uint32_t row_size_bytes = get_arg(args::row_size_bytes);
    uint32_t batch_size_bytes = get_arg(args::batch_size_bytes);

#ifdef REDUCE_SCALER
    const uint32_t scaler_bits = get_arg(args::scaler_bits);
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);
#endif

    // Emit tiles in N, W_skip, H, W_chunk order to match the chunked iteration of the
    // unified reduce compute kernel (row_chunk = DEST_AUTO_LIMIT). For shard_Wt=1 this
    // degenerates to one column per chunk; for shard_Wt>1 it interleaves columns.
    // Int32 SFPU max reserves one DST for the binary-fold work tile (DEST_AUTO_LIMIT - 1).
    // Accurate fp32: the host sets enable_fp32_sfpu so SFPU chunk sizing here matches the
    // compute kernel.
    constexpr auto fp32_mode = get_arg(args::enable_fp32_sfpu) != 0 ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;
    // The data format has to be a constant expression here (it is a template argument below), so it
    // is read with the free function rather than off a DataflowBuffer object: DataflowBuffer's
    // constructor is not constexpr, so no such object is usable in a constant expression.
    constexpr DataFormat reduce_format = get_dataformat(dfb::in0);
    constexpr bool use_sfpu_reduce_path = is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format, fp32_mode>();
    constexpr uint32_t row_chunk =
        use_sfpu_reduce_path ? (compute_kernel_lib::DEST_AUTO_LIMIT - 1) : compute_kernel_lib::DEST_AUTO_LIMIT;

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

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t Wt = get_arg_val<uint32_t>(1);
    uint32_t Ht = get_arg_val<uint32_t>(2);
    uint32_t batch = get_arg_val<uint32_t>(3);
    uint32_t row_size_bytes = get_arg_val<uint32_t>(4);
    uint32_t batch_size_bytes = get_arg_val<uint32_t>(5);

    constexpr uint32_t dfb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t dfb_id_in1 = get_compile_time_arg_val(1);

#ifdef REDUCE_SCALER
    constexpr uint32_t dfb_id_in2 = get_compile_time_arg_val(2);
    constexpr uint32_t scaler_bits = get_compile_time_arg_val(3);
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb_id_in2, REDUCE_OP, REDUCE_DIM>(scaler_f);
#endif

    // Emit tiles in N, W_skip, H, W_chunk order to match the chunked iteration of the
    // unified reduce compute kernel (row_chunk = DEST_AUTO_LIMIT). For shard_Wt=1 this
    // degenerates to one column per chunk; for shard_Wt>1 it interleaves columns.
    // Int32 SFPU max reserves one DST for the binary-fold work tile (DEST_AUTO_LIMIT - 1).
    // Accurate fp32 mean: host sets CT arg 4 to 1 so SFPU chunk sizing here matches the compute kernel.
    constexpr auto fp32_mode = get_compile_time_arg_val(4) != 0 ? ReduceFp32Mode::Accurate : ReduceFp32Mode::Fast;
    constexpr DataFormat reduce_format = get_dataformat(dfb_id_in0);
    constexpr bool use_sfpu_reduce_path = is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format, fp32_mode>();
    constexpr uint32_t row_chunk =
        use_sfpu_reduce_path ? (compute_kernel_lib::DEST_AUTO_LIMIT - 1) : compute_kernel_lib::DEST_AUTO_LIMIT;

    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(dfb_id_in0);

    Noc noc;
    DataflowBuffer dfb_in0(dfb_id_in0);
    DataflowBuffer dfb_in1(dfb_id_in1);

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

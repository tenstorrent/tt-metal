// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    std::uint32_t src_addr = get_arg_val<std::uint32_t>(0);
    std::uint32_t num_tiles = get_arg_val<std::uint32_t>(1);
    std::uint32_t start_id = get_arg_val<std::uint32_t>(2);

    constexpr std::uint32_t scaler_bits = get_compile_time_arg_val(0);
    constexpr std::uint32_t Wt = get_compile_time_arg_val(1);
    constexpr auto tensor_args = TensorAccessorArgs<2>();

    constexpr std::uint32_t cb_id_in2 = tt::CBIndex::c_2;
    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2, REDUCE_OP, REDUCE_DIM>(scaler_f);

    constexpr std::uint32_t cb_id_in0 = tt::CBIndex::c_0;
    constexpr std::uint32_t onetile = 1;
    const std::uint32_t tile_bytes = get_tile_size(cb_id_in0);

    auto tensor_accessor = TensorAccessor(tensor_args, src_addr);
    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);

    const std::uint32_t num_rows = num_tiles / Wt;
    for (std::uint32_t row = 0; row < num_rows; ++row) {
        const std::uint32_t row_start_id = start_id + row * Wt;
#ifdef WELFORD_TWO_PASS_L1_REPLAY
        // The compute kernel keeps this complete row in the enlarged input CB and
        // indexes it twice, so DRAM is traversed only once.
        constexpr std::uint32_t num_passes = 1;
#else
#ifdef WELFORD_TWO_PASS_BFP8_INPUT
        constexpr std::uint32_t num_front_retained = 2;
#else
        constexpr std::uint32_t num_front_retained = 3;
#endif
        constexpr std::uint32_t num_passes = Wt <= num_front_retained + 1 ? 1 : 2;
#endif
        for (std::uint32_t pass = 0; pass < num_passes; ++pass) {
#ifdef WELFORD_TWO_PASS_L1_REPLAY
            constexpr std::uint32_t pass_start = 0;
            constexpr std::uint32_t pass_end = Wt;
#else
            // Compute retains the first two or three transposed tiles and the final
            // tile in DEST across passes, so only stream the middle tiles on
            // pass two. Tile order remains unchanged.
            const std::uint32_t pass_start = pass == 0 ? 0 : std::min(Wt, num_front_retained);
            const std::uint32_t pass_end = pass == 0 ? Wt : Wt - 1;
#endif
            for (std::uint32_t wt = pass_start; wt < pass_end; ++wt) {
                cb_in0.reserve_back(onetile);
                noc.async_read(
                    tensor_accessor, cb_in0, tile_bytes, {.page_id = row_start_id + wt}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_in0.push_back(onetile);
            }
        }
    }
}

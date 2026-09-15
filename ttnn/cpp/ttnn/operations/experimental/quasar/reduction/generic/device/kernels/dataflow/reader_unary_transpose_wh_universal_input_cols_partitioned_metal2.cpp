// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of reader_unary_transpose_wh_universal_input_cols_partitioned.cpp (interleaved H/W
// reduce reader). Identical dataflow: fill the reduce-scaler DFB once, then stream input tiles into
// the input DFB in the chunked N/W_skip/H/W_chunk order. CB indices → dfb:: bindings, source
// TensorAccessor → tensor:: binding (src_addr runtime arg gone), CTAs/RTAs named. Legacy retained for
// not-yet-ported reduce paths.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t col_start_tile_id = get_arg(args::col_start_tile_id);
    uint32_t curr_col_in_batch = get_arg(args::curr_col_in_batch);
    uint32_t num_cols = get_arg(args::num_cols);

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t scaler_bits = get_arg(args::scaler_bits);
    constexpr bool use_welford = get_arg(args::use_welford) != 0;

    constexpr DataFormat reduce_format = get_dataformat(dfb::in);
    constexpr bool use_sfpu_reduce_path = is_sfpu_reduce_path<REDUCE_OP, REDUCE_DIM, reduce_format>();
    constexpr uint32_t row_chunk = use_welford ? 1
                                               : (use_sfpu_reduce_path ? (compute_kernel_lib::DEST_AUTO_LIMIT - 1)
                                                                       : compute_kernel_lib::DEST_AUTO_LIMIT);

    constexpr uint32_t onetile = 1;

    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler, REDUCE_OP, REDUCE_DIM>(scaler_f);

    const auto tensor_accessor = TensorAccessor(tensor::input);

    Noc noc;
    DataflowBuffer cb_in0(dfb::in);
    const uint32_t tile_bytes = cb_in0.get_entry_size();

    uint32_t w = curr_col_in_batch;

    for (uint32_t i = 0; i < num_cols; i += row_chunk) {
        uint32_t chunk_end = std::min(i + row_chunk, num_cols);
        uint32_t curr_id = col_start_tile_id;
        uint32_t reset_curr_id = curr_id;
        uint32_t reset_w = w;
        uint32_t reset_col_start = col_start_tile_id;

        for (uint32_t j = 0; j < Ht; ++j) {
            w = reset_w;
            col_start_tile_id = reset_col_start;
            for (uint32_t k = i; k < chunk_end; ++k) {
                cb_in0.reserve_back(onetile);
                noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = curr_id}, {.offset_bytes = 0});
                noc.async_read_barrier();
                cb_in0.push_back(onetile);

                ++w;

                if (w == Wt) {
                    col_start_tile_id = curr_id + (Ht - j - 1) * Wt + 1;
                    curr_id = col_start_tile_id + j * Wt;
                    w = 0;
                } else {
                    ++curr_id;
                    ++col_start_tile_id;
                }
            }
            curr_id = reset_curr_id + (j + 1) * Wt;  // stride in H
        }
    }
}

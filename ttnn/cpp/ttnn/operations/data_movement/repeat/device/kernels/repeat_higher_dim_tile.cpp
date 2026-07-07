// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// TILE higher-dim repeat; direct per-page transfer (tile pages are atomic on one core).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

using namespace tt::data_movement::common;

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_arg_val<uint32_t>(1);
    const uint32_t higher_dim_start = get_arg_val<uint32_t>(2);
    const uint32_t higher_dim_end = get_arg_val<uint32_t>(3);
    const uint32_t lower_dim_start = get_arg_val<uint32_t>(4);
    const uint32_t lower_dim_end = get_arg_val<uint32_t>(5);
    const uint32_t repetitions = get_arg_val<uint32_t>(6);
    const uint32_t nop = get_arg_val<uint32_t>(7);

    constexpr uint32_t original_page_size_bytes = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t LOWER_DIMS = get_compile_time_arg_val(2);
    constexpr uint32_t REP_DIM = get_compile_time_arg_val(3);
    constexpr auto src_args = TensorAccessorArgs<4, 0>();
    constexpr auto dst_args =
        TensorAccessorArgs<src_args.next_compile_time_args_offset(), src_args.num_common_runtime_args()>();

    constexpr uint32_t LOWER_DIMS_TIMES_REP_DIM = LOWER_DIMS * REP_DIM;

    if (nop == 1) {
        return;
    }

    const auto s = TensorAccessor(src_args, src_addr);
    const auto d = TensorAccessor(dst_args, dst_addr);

    Noc noc;
    CircularBuffer cb(cb_id_in0);
    cb.reserve_back(1);
    const uint32_t cb_slot = cb.get_write_ptr();
    cb.push_back(1);
    const CoreLocalMem<uint32_t> cb_mem(cb_slot);

    for (uint32_t h = higher_dim_start; h < higher_dim_end; h++) {
        const uint32_t h_offset = h * LOWER_DIMS_TIMES_REP_DIM;
        const uint32_t h_offset_rep = h_offset * repetitions;
        for (uint32_t r = 0; r < REP_DIM; r++) {
            const uint32_t r_offset = r * LOWER_DIMS;
            for (uint32_t l = lower_dim_start; l < lower_dim_end; l++) {
                const uint32_t read_offset = h_offset + r_offset + l;
                noc.async_read(
                    s,
                    cb_mem,
                    original_page_size_bytes,
                    {.page_id = read_offset, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();
                for (uint32_t n = 0; n < repetitions; n++) {
                    const uint32_t write_offset = h_offset_rep + n * LOWER_DIMS_TIMES_REP_DIM + r_offset + l;
                    noc.async_write(
                        cb_mem,
                        d,
                        original_page_size_bytes,
                        {.offset_bytes = 0},
                        {.page_id = write_offset, .offset_bytes = 0});
                }
                noc.async_write_barrier();
            }
        }
    }
}

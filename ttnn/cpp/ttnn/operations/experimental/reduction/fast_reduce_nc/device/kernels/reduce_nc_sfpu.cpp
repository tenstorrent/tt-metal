// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Float32 and Int32 variant of reduce_nc.cpp: tiles are unpacked to DEST and summed on the SFPU,
// so Float32 keeps full precision (the FPU path truncates it to tf32) and Int32 is supported at all.

#include "api/compute/common.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

void kernel_main() {
    constexpr auto num_output_tiles = get_arg(args::num_output_tiles);
    constexpr auto num_input_tiles = get_arg(args::num_input_tiles);
    constexpr auto input_granularity = get_arg(args::input_granularity);

    constexpr uint32_t onetile = 1;
    constexpr uint32_t dst_acc = 0;
    constexpr uint32_t dst_work = 1;
    constexpr uint32_t num_input_tiles_iter = num_input_tiles / input_granularity;
    constexpr DataFormat format = static_cast<DataFormat>(unpack_src_format[dfb::in0]);

    DataflowBuffer cb_in0_obj(dfb::in0);
    DataflowBuffer cb_in1_obj(dfb::in1);
    DataflowBuffer cb_out0_obj(dfb::out0);

    compute_kernel_hw_startup(dfb::in0, dfb::out0);
    copy_init(dfb::in0);
    compute_kernel_lib::detail::sfpu_reduce_fold_init<PoolType::SUM, format>();
    // The reader always produces the FPU path's zero tile; it is unused here.
    cb_in1_obj.wait_front(onetile);

    for (uint32_t i = 0; i < num_output_tiles; i++) {
        tile_regs_acquire();
        for (uint32_t j = 0; j < num_input_tiles_iter; ++j) {
            cb_in0_obj.wait_front(input_granularity);
            for (uint32_t k = 0; k < input_granularity; k++) {
                compute_kernel_lib::detail::sfpu_copy_and_fold<PoolType::SUM, format>(
                    dfb::in0, k, dst_acc, dst_work, j == 0 && k == 0);
            }
            cb_in0_obj.pop_front(input_granularity);
        }
        tile_regs_commit();
        cb_out0_obj.reserve_back(onetile);
        tile_regs_wait();
        pack_tile(dst_acc, dfb::out0);
        tile_regs_release();
        cb_out0_obj.push_back(onetile);
    }
    cb_in1_obj.pop_front(onetile);
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/compute/tile_move_copy.h"
#include "api/compute/matmul.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tilize.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "experimental/kernel_args.h"

using std::uint32_t;

// matmul C=A*B using dims MK*KN = MN (row major order)
//
void kernel_main() {
    constexpr uint32_t onetile = 1;

    constexpr uint32_t transpose_hw = get_arg(args::transpose_hw);
    uint32_t batch = get_arg(args::batch);
    uint32_t Mt = get_arg(args::Mt);
    uint32_t Kt = get_arg(args::Kt);
    uint32_t Nt = get_arg(args::Nt);

    DataflowBuffer dfb_in0(dfb::in0);
    DataflowBuffer dfb_in1(dfb::in1);
    DataflowBuffer dfb_intermed0(dfb::intermed0);
    DataflowBuffer dfb_intermed1(dfb::intermed1);
    DataflowBuffer dfb_intermed2(dfb::intermed2);
    DataflowBuffer dfb_out(dfb::out);

    constexpr uint32_t num_rows_in_one_tile = 32;

    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb::in0, dfb::in1, dfb::intermed0);
    matmul_init(dfb::in0, dfb::in1, transpose_hw);

    for (uint32_t nb = 0; nb < batch; ++nb) {
        for (uint32_t mt_C = 0; mt_C < Mt; ++mt_C) {    // output tile of C
            for (uint32_t nt_C = 0; nt_C < Nt; ++nt_C)  // output tile index of C
            {
                for (uint32_t tile_row_id = 0; tile_row_id < num_rows_in_one_tile; ++tile_row_id) {
                    tile_regs_acquire();
                    for (uint32_t kt = 0; kt < Kt; ++kt) {
                        if (tile_row_id == 0) {
                            dfb_in0.wait_front(kt + 1);
                        }
                        dfb_in1.wait_front(onetile);

                        matmul_tiles(dfb::in0, dfb::in1, kt, 0, 0);

                        dfb_in1.pop_front(onetile);
                    }
                    tile_regs_commit();

                    dfb_intermed0.reserve_back(onetile);
                    tile_regs_wait();
                    pack_tile(0, dfb::intermed0);
                    tile_regs_release();
                    dfb_intermed0.push_back(onetile);

                    // untilize tile and write to intermed1 with reconfiguration
                    compute_kernel_lib::untilize<
                        onetile,
                        dfb::intermed0,
                        dfb::intermed1,
                        compute_kernel_lib::untilize_config::InitUninitMode::InitAndUninit,
                        compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                        compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::UnpackReconfigure>(1);

                    reconfig_data_format_srca(dfb::intermed0, dfb::in1);
                    matmul_init(dfb::in0, dfb::in1, transpose_hw);
                }
                dfb_in0.pop_front(Kt);

                // intermed2 comes from reader; untilized row-major tile
                // tilize intermed2 and write to out with reconfiguration
                compute_kernel_lib::tilize<onetile, dfb::intermed2, dfb::out>(1);

                pack_reconfig_data_format(dfb::out, dfb::intermed0);
                reconfig_data_format(dfb::intermed2, dfb::in1, dfb::intermed2, dfb::in0);
                matmul_block_init(dfb::in0, dfb::in1, transpose_hw);
            }
        }
    }
}

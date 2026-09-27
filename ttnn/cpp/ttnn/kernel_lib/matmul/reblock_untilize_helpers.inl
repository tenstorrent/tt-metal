// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/compute/pack_untilize.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "internal/mod_div_lib.h"
#include "api/dataflow/dataflow_buffer.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

namespace compute_kernel_lib {

template <uint32_t OutSubblockW, uint32_t OutBlockW, bool Reconfigure>
inline void reblock_and_untilize(
    uint32_t in0_num_subblocks, uint32_t out_subblock_h, uint32_t interm_cb_id, uint32_t out_cb_id) {
    static_assert(OutSubblockW > 0 && OutBlockW > 0 && OutBlockW % OutSubblockW == 0);
    constexpr uint32_t num_subblocks_w = OutBlockW / OutSubblockW;
    DataflowBuffer interm_buf(interm_cb_id), out_buf(out_cb_id);

    if constexpr (Reconfigure) {
        reconfig_data_format_srca(interm_cb_id);
        pack_reconfig_data_format(out_cb_id);
    }
    pack_untilize_dest_init<OutSubblockW, OutBlockW>(out_cb_id);
    copy_init(interm_cb_id);

    const uint32_t out_subblock_num_tiles = out_subblock_h * OutSubblockW;
    const uint32_t num_tiles_in_row_of_subblocks = mulsi3(out_subblock_num_tiles, num_subblocks_w);

    for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; in0_subblock++) {
        interm_buf.wait_front(num_tiles_in_row_of_subblocks);

        uint32_t within_block_index = 0;
        for (uint32_t h = 0; h < out_subblock_h; h++) {
            uint32_t block_offset = 0;
            out_buf.reserve_back(OutBlockW);
            for (uint32_t n = 0; n < num_subblocks_w; n++) {
                tile_regs_acquire();
                for (uint32_t w = 0; w < OutSubblockW; w++) {
                    copy_tile(interm_cb_id, block_offset + within_block_index + w, w);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<OutSubblockW, OutBlockW>(out_cb_id, 1, n);
                tile_regs_release();
                block_offset += out_subblock_num_tiles;
            }
            out_buf.push_back(OutBlockW);
            within_block_index += OutSubblockW;
        }
        interm_buf.pop_front(num_tiles_in_row_of_subblocks);
    }

    pack_untilize_uninit(interm_cb_id);
}

}  // namespace compute_kernel_lib

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DRAM DOWNLOAD FLOOR — writer half (BRISC / NOC_1), the op's own split: W_up + W_down.
//
// The split is not cosmetic. `DM_DEDICATED_NOC` binds the reader to NOC_0 and the writer to NOC_1,
// so "which kernel issues the read" IS the NoC choice, and the real op deliberately puts W_up and
// W_down on the writer so the two weight streams use both networks' injection ports. A one-kernel
// download bench would measure a single NoC and understate the achievable rate.
//
// See dl_reader.cpp for why there is exactly one barrier.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t READ_WU = get_compile_time_arg_val(0);
constexpr uint32_t READ_WD = get_compile_time_arg_val(1);
constexpr uint32_t HID_T = get_compile_time_arg_val(2);      // hidden tiles (W_up row pitch)
constexpr uint32_t EMB_T = get_compile_time_arg_val(3);      // emb tiles   (W_down row pitch)
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(4);  // 576

constexpr uint32_t TA_BASE = 5;
constexpr auto wu_args = TensorAccessorArgs<TA_BASE>();
constexpr auto wd_args = TensorAccessorArgs<wu_args.next_compile_time_args_offset()>();

constexpr uint32_t CB_WU = 1;
constexpr uint32_t CB_WD = 2;

void kernel_main() {
    const uint32_t wu_addr = get_arg_val<uint32_t>(0);
    const uint32_t wd_addr = get_arg_val<uint32_t>(1);
    const uint32_t kstart = get_arg_val<uint32_t>(2);  // my grid row's first emb K-tile (gate/up side)
    const uint32_t kr = get_arg_val<uint32_t>(3);
    const uint32_t hstart = get_arg_val<uint32_t>(4);  // my grid column's first hidden tile
    const uint32_t hn = get_arg_val<uint32_t>(5);
    const uint32_t ecstart = get_arg_val<uint32_t>(6);  // my first emb OUTPUT tile (W_down side)
    const uint32_t ec = get_arg_val<uint32_t>(7);       // my emb-output tile count

    if constexpr (READ_WU) {
        const auto wu_acc = TensorAccessor(wu_args, wu_addr, BFP4_TILE);
        cb_reserve_back(CB_WU, kr * hn);
        const uint32_t wp = get_write_ptr(CB_WU);
        for (uint32_t k = 0; k < kr; ++k) {
            noc_async_read(wu_acc.get_noc_addr((kstart + k) * HID_T + hstart), wp + k * hn * BFP4_TILE, hn * BFP4_TILE);
        }
    }

    if constexpr (READ_WD) {
        // W_down is [hidden, emb]: the emb OUTPUT axis is split across ALL cores (`ec` tiles each),
        // and every core contracts over the WHOLE hidden axis, so it reads all HID_T rows of its own
        // narrow emb slice. That is why W_down's request (ec*576) is smaller than gate/up's (hn*576)
        // and its request COUNT is larger — the asymmetry is in the op, not in this bench.
        const auto wd_acc = TensorAccessor(wd_args, wd_addr, BFP4_TILE);
        cb_reserve_back(CB_WD, HID_T * ec);
        const uint32_t wp = get_write_ptr(CB_WD);
        for (uint32_t j = 0; j < HID_T; ++j) {
            noc_async_read(wd_acc.get_noc_addr(j * EMB_T + ecstart), wp + j * ec * BFP4_TILE, ec * BFP4_TILE);
        }
    }

    noc_async_read_barrier();
}

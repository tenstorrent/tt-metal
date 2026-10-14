// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "ttnn/operations/data_movement/common/kernels/common.hpp"

// Band assembler + output writer.
//
// Per band, cb_rm holds one untilized 32x32 block per input tile (pushed input-major, in read
// order). Each block's row r starts at r * tile_row_bytes. This kernel gathers the logical
// (unpadded) row segments of every input into one contiguous row-major band in cb_asm -- an
// arbitrary-byte-offset copy no NoC or LLK path can do, hence the CPU memmove -- zero-fills the
// output's width padding, and after the compute kernel retilizes the band, writes the output
// tiles to the interleaved output buffer.
void kernel_main() {
    constexpr uint32_t cb_rm = get_compile_time_arg_val(0);
    constexpr uint32_t cb_asm = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t num_tensors = get_compile_time_arg_val(3);
    constexpr uint32_t out_wt = get_compile_time_arg_val(4);
    constexpr uint32_t total_in_wt = get_compile_time_arg_val(5);
    constexpr uint32_t tile_row_bytes = get_compile_time_arg_val(6);
    constexpr uint32_t out_row_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t tail_bytes = get_compile_time_arg_val(8);
    constexpr auto dst_args = TensorAccessorArgs<9>();

    constexpr uint32_t tile_height = 32;

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_bands = get_arg_val<uint32_t>(1);
    const uint32_t band_start = get_arg_val<uint32_t>(2);

    uint32_t wt[num_tensors];
    uint32_t w_bytes[num_tensors];
    tt_l1_ptr uint32_t* arg_ptr = (tt_l1_ptr uint32_t*)get_arg_addr(3);
    for (uint32_t i = 0; i < num_tensors; ++i) {
        wt[i] = arg_ptr[i];
        w_bytes[i] = arg_ptr[num_tensors + i];
    }

    const auto s = TensorAccessor(dst_args, dst_addr);
    const uint32_t rm_page_bytes = get_tile_size(cb_rm);
    const uint32_t out_tile_bytes = get_tile_size(cb_out);

    DataflowBuffer dfb_rm(cb_rm);
    DataflowBuffer dfb_asm(cb_asm);
    DataflowBuffer dfb_out(cb_out);
    Noc noc;

    for (uint32_t b = 0; b < num_bands; ++b) {
        const uint32_t band = band_start + b;

        // cb_rm capacity is exactly one band, so the band always starts at the CB base and is
        // contiguous; same for cb_asm.
        dfb_rm.wait_front(total_in_wt);
        dfb_asm.reserve_back(out_wt);
        const uint32_t src_base = dfb_rm.get_read_ptr();
        const uint32_t dst_base = dfb_asm.get_write_ptr();

        uint32_t col_bytes = 0;
        uint32_t tile_idx = 0;
        for (uint32_t i = 0; i < num_tensors; ++i) {
            uint32_t remaining = w_bytes[i];
            for (uint32_t t = 0; t < wt[i]; ++t) {
                const uint32_t bytes = remaining < tile_row_bytes ? remaining : tile_row_bytes;
                const uint32_t src_block = src_base + tile_idx * rm_page_bytes;
                const uint32_t dst_col = dst_base + col_bytes + t * tile_row_bytes;
                for (uint32_t r = 0; r < tile_height; ++r) {
                    tt::data_movement::common::tt_memmove<false, false, false, tile_row_bytes>(
                        noc, dst_col + r * out_row_bytes, src_block + r * tile_row_bytes, bytes);
                }
                remaining -= bytes;
                ++tile_idx;
            }
            col_bytes += w_bytes[i];
        }

        if constexpr (tail_bytes > 0) {
            for (uint32_t r = 0; r < tile_height; ++r) {
                volatile tt_l1_ptr uint16_t* tail =
                    reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dst_base + col_bytes + r * out_row_bytes);
                for (uint32_t k = 0; k < tail_bytes / 2; ++k) {
                    tail[k] = 0;
                }
            }
            // Unlike tt_memmove above, these tail stores don't self-drain. Since L1 write-requests
            // from one client are processed in order, a blocking load of the last-written word
            // guarantees all tail stores landed before push_back publishes the band to the compute
            // kernel (same contract as tt_memmove's !copy_async drain in common.hpp).
            volatile tt_l1_ptr uint32_t* drain_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                (dst_base + col_bytes + (tile_height - 1) * out_row_bytes + tail_bytes - 1) & ~uint32_t{3});
#if defined(ARCH_QUASAR)
            // Quasar has no ckernel::load_blocking; this factory is gated off on Quasar at host
            // (see can_use_tiled_unaligned_concat), so this branch only keeps the kernel compiling.
            (void)*drain_ptr;
#else
            (void)ckernel::load_blocking(drain_ptr);
#endif
        }

        dfb_asm.push_back(out_wt);
        dfb_rm.pop_front(total_in_wt);

        const uint32_t band_first_tile = band * out_wt;
        for (uint32_t t = 0; t < out_wt; ++t) {
            dfb_out.wait_front(1);
            noc.async_write(dfb_out, s, out_tile_bytes, {}, {.page_id = band_first_tile + t});
            noc.async_writes_flushed();
            dfb_out.pop_front(1);
        }
    }
    noc.async_write_barrier();
}

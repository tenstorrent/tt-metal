// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared writer kernel for fused multi-scale deformable attention. Used
// unchanged by both readers.
//
// Each block from compute carries up to 32 accumulated query outputs stacked
// vertically across N_D_TILES tiles laid side by side (32 value-channels per
// tile). For each block the writer gathers a query row's D values across the
// d-tiles into one stick and NoC-writes it into its place in the output.
//
// The output is (B, Q, H*D), so query (b, q) is one page of H*D bf16 and head
// `head` occupies bytes [head*D*2, (head+1)*D*2) of it. Writing at that offset
// is what concatenates the heads — no post-op reshape or permute, and no
// cross-core communication, because each block owns exactly one head. D being a
// multiple of 16 keeps the offset 32-B aligned.
//
// Nothing model-specific happens here: no SCA scatter, no rebatch, no residual.
//
// With SPLIT_GATHER the writer also gathers rows [SPLIT_ROW, 32) of every
// point the reader stages (msda_gather_rows.hpp): it waits for the reader's
// `ready` count, gathers from the point it posted to the mailbox, and bumps
// `done`. The data-movement RISC here is otherwise idle for all but the last
// few microseconds of each block, and it issues its reads on its own NoC.
//
// Tile face layout (bf16, 32x32 = 4 faces of 16x16, 2048 B) is documented in
// ../msda_tile_layout.hpp. D-tile k holds value columns [k*32, k*32+31].
//
// Per-tile runtime args (3 per tile): (base_page, head, v_rows), where
// base_page = b * Q + q_start; rows 0..v_rows-1 go to consecutive pages.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/msda_gather_rows.hpp"

constexpr uint32_t output_tile_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t output_scratch_cb_index = get_compile_time_arg_val(1);
constexpr uint32_t output_page_nbytes = get_compile_time_arg_val(2);  // aligned H*D*2
constexpr uint32_t D = get_compile_time_arg_val(3);
constexpr bool SPLIT_GATHER = get_compile_time_arg_val(4) != 0;
constexpr uint32_t gather_ready_sem_id = get_compile_time_arg_val(5);
constexpr uint32_t gather_done_sem_id = get_compile_time_arg_val(6);
constexpr uint32_t gather_mailbox_cb_index = get_compile_time_arg_val(7);
constexpr uint32_t POINTS_PER_BLOCK = get_compile_time_arg_val(8);
constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(9);
constexpr uint32_t NUM_KEYS = get_compile_time_arg_val(10);
constexpr bool VALUE_PACKED = get_compile_time_arg_val(11) != 0;
constexpr uint32_t value_stick_nbytes = get_compile_time_arg_val(12);
constexpr uint32_t value_page_nbytes = get_compile_time_arg_val(13);

constexpr auto output_args = TensorAccessorArgs<14>();
constexpr auto value_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();

struct WriterGatherCfg {
    static constexpr uint32_t D = ::D;
    static constexpr uint32_t NUM_KEYS = ::NUM_KEYS;
    static constexpr uint32_t NUM_HEADS = ::NUM_HEADS;
    static constexpr bool VALUE_PACKED = ::VALUE_PACKED;
    static constexpr uint32_t STICK_NBYTES = value_stick_nbytes;
};

constexpr uint32_t HALF_STICK_NBYTES = 32;
constexpr uint32_t HALF_WORDS = HALF_STICK_NBYTES / sizeof(uint32_t);
constexpr uint32_t TILE_NBYTES = fused_msda_tile_layout::TILE_NBYTES;

// Derived from D, not from a padded byte count.
constexpr uint32_t STICK_WORDS = D / 2;
constexpr uint32_t WORDS_PER_TILE_ROW = 2 * HALF_WORDS;
constexpr uint32_t N_D_TILES = (STICK_WORDS + WORDS_PER_TILE_ROW - 1) / WORDS_PER_TILE_ROW;
constexpr uint32_t HEAD_NBYTES = D * sizeof(uint16_t);
static_assert(D % 16 == 0 && D > 0, "D must be a positive multiple of 16");

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    const uint32_t value_addr = get_arg_val<uint32_t>(2);

    const auto output_acc = TensorAccessor(output_args, output_addr, output_page_nbytes);
    [[maybe_unused]] const auto value_acc = TensorAccessor(value_args, value_addr, value_page_nbytes);

    Noc noc;
    CircularBuffer output_tile_cb(output_tile_cb_index);
    CircularBuffer output_scratch_cb(output_scratch_cb_index);

    output_scratch_cb.reserve_back(1);
    const uint32_t scratch_l1 = output_scratch_cb.get_write_ptr();

    [[maybe_unused]] const uint32_t gather_mailbox_l1 =
        SPLIT_GATHER ? CircularBuffer(gather_mailbox_cb_index).get_write_ptr() : 0u;
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* gather_ready =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather_ready_sem_id));
    [[maybe_unused]] volatile tt_l1_ptr uint32_t* gather_done =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(gather_done_sem_id));
    [[maybe_unused]] uint32_t gather_seq = 0;

    uint32_t arg_idx = 3;
    for (uint32_t t = 0; t < num_output_tiles; ++t) {
        const uint32_t base_page = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t head = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v_rows = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t head_byte_offset = head * HEAD_NBYTES;

        if constexpr (SPLIT_GATHER) {
            // The block's output cannot exist before its last point is
            // gathered, so helping with every point first costs no overlap.
            for (uint32_t j = 0; j < POINTS_PER_BLOCK; ++j) {
                ++gather_seq;
                while (*gather_ready != gather_seq) {
                }
                const auto pt = fused_msda_gather::fetch_point(gather_mailbox_l1);
                fused_msda_gather::gather_rows<WriterGatherCfg>(
                    noc, value_acc, pt, fused_msda_gather::SPLIT_ROW, fused_msda_tile_layout::TILE_MAX_ROWS);
                noc.async_read_barrier();
                // Plain zero stores above; keep them ahead of the release.
                asm volatile("" ::: "memory");
                *gather_done = gather_seq;
            }
        }

        output_tile_cb.wait_front(N_D_TILES);
        const uint32_t tile_l1 = output_tile_cb.get_read_ptr();

        for (uint32_t r = 0; r < v_rows; ++r) {
            const auto off = fused_msda_tile_layout::tile_row_offsets(r);
            CoreLocalMem<volatile uint32_t> dst(scratch_l1);
            for (uint32_t k = 0; k < N_D_TILES; ++k) {
                const uint32_t base = k * WORDS_PER_TILE_ROW;
                const uint32_t words_k =
                    (STICK_WORDS - base < WORDS_PER_TILE_ROW) ? (STICK_WORDS - base) : WORDS_PER_TILE_ROW;
                const uint32_t lo_words = words_k < HALF_WORDS ? words_k : HALF_WORDS;
                const uint32_t hi_words = words_k - lo_words;
                const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
                CoreLocalMem<volatile uint32_t> sl(ktile_l1 + off.lo);
                CoreLocalMem<volatile uint32_t> sh(ktile_l1 + off.hi);
                for (uint32_t i = 0; i < lo_words; ++i) {
                    dst[base + i] = sl[i];
                }
                for (uint32_t i = 0; i < hi_words; ++i) {
                    dst[base + HALF_WORDS + i] = sh[i];
                }
            }

            CoreLocalMem<uint32_t> src(scratch_l1);
            noc.async_write(
                src,
                output_acc,
                HEAD_NBYTES,
                {.offset_bytes = 0},
                {.page_id = base_page + r, .offset_bytes = head_byte_offset});
            noc.async_writes_flushed();
        }
        noc.async_write_barrier();
        output_tile_cb.pop_front(N_D_TILES);
    }
}

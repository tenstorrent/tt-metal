// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Shared writer kernel for fused multi-scale deformable attention. Used
// unchanged by both readers.
//
// Each block from compute carries up to 32 accumulated query outputs stacked
// vertically across N_D_TILES tiles laid side by side (32 value-channels per
// tile). For each block the writer moves each query row's D values into its
// place in the output.
//
// A tile row is two 16-column halves in different faces, 32 B each. The writer
// NoC-writes each half straight from the tile to its place in the output page,
// so no RISC copies the block.
//
// The output is (B, Q, H*D), so query (b, q) is one page of H*D bf16 and head
// `head` occupies bytes [head*D*2, (head+1)*D*2) of it. Writing at that offset
// is what concatenates the heads — no post-op reshape or permute, and no
// cross-core communication, because each block owns exactly one head. Every
// write lands at the page base plus a multiple of HALF_NBYTES, and the
// static_asserts below hold HALF_NBYTES against the NoC write alignment of L1
// and DRAM, so the output takes any H at any D % 16 == 0. Packed value reads
// have their own constraint, checked in the program factory.
//
// With SPLIT_GATHER the writer also gathers part of every row-major block the
// reader reserves; see msda_gather_rows.hpp.
//
// Tile face layout (bf16, 32x32 = 4 faces of 16x16, 2048 B) is documented in
// ../msda_tile_layout.hpp. D-tile k holds value columns [k*32, k*32+31].
//
// Runtime args:
//   [0]        output buffer address
//   [1]        num_output_tiles
//   [2]        value buffer address (read only with SPLIT_GATHER)
//   [3 ...]    per output tile: base_page, head, v_rows, where
//              base_page = b * Q + q_start; rows 0..v_rows-1 go to consecutive pages
// With MSDA_DEVICE_COUNTS defined, [1] is the address of the (MSDA_BATCH,)
// uint32 valid query counts and [3], [4] are this core's index and the core
// count. The writer stages its own copy of the counts in MSDA_WRITER_COUNTS_CB
// and derives the same tile run as the reader (msda_device_counts.hpp);
// MSDA_NUM_QUERIES is Q.
// Compile-time arg order is fixed by fused_msda_program_factory.cpp (writer_ct).

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/msda_tile_layout.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/msda_device_counts.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/fused_msda/device/kernels/dataflow/msda_gather_rows.hpp"

constexpr uint32_t output_tile_cb_index = get_compile_time_arg_val(0);
constexpr uint32_t output_page_nbytes = get_compile_time_arg_val(1);  // aligned H*D*2
constexpr uint32_t D = get_compile_time_arg_val(2);
constexpr bool SPLIT_GATHER = get_compile_time_arg_val(3) != 0;
constexpr uint32_t gather_ready_sem_id = get_compile_time_arg_val(4);
constexpr uint32_t gather_done_sem_id = get_compile_time_arg_val(5);
constexpr uint32_t gather_mailbox_cb_index = get_compile_time_arg_val(6);
constexpr uint32_t gather_mailbox_nbytes = get_compile_time_arg_val(7);  // allocated page; 0 without SPLIT_GATHER
constexpr uint32_t POINTS_PER_BLOCK = get_compile_time_arg_val(8);
constexpr uint32_t NUM_HEADS = get_compile_time_arg_val(9);
constexpr uint32_t NUM_KEYS = get_compile_time_arg_val(10);
constexpr auto VALUE_LAYOUT = static_cast<fused_msda_value_layout::ValueLayout>(get_compile_time_arg_val(11));
constexpr uint32_t value_stick_nbytes = get_compile_time_arg_val(12);
constexpr uint32_t value_page_nbytes = get_compile_time_arg_val(13);

constexpr auto output_args = TensorAccessorArgs<14>();
constexpr auto value_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
#ifdef MSDA_DEVICE_COUNTS
constexpr auto counts_args = TensorAccessorArgs<value_args.next_compile_time_args_offset()>();
#endif

struct WriterGatherCfg {
    static constexpr uint32_t D = ::D;
    static constexpr uint32_t NUM_KEYS = ::NUM_KEYS;
    static constexpr uint32_t NUM_HEADS = ::NUM_HEADS;
    static constexpr fused_msda_value_layout::ValueLayout VALUE_LAYOUT = ::VALUE_LAYOUT;
    static constexpr uint32_t STICK_NBYTES = value_stick_nbytes;
};

constexpr uint32_t HALF_COLS = fused_msda_tile_layout::FACE_ROW_COLS;
constexpr uint32_t HALF_NBYTES = fused_msda_tile_layout::FACE_ROW_NBYTES;
constexpr uint32_t TILE_NBYTES = fused_msda_tile_layout::TILE_NBYTES;
constexpr uint32_t TILE_COLS = 2 * HALF_COLS;
constexpr uint32_t N_D_TILES = (D + TILE_COLS - 1) / TILE_COLS;
constexpr uint32_t HEAD_NBYTES = D * sizeof(uint16_t);
// Each half is written whole, so D must end on a half boundary.
static_assert(D % HALF_COLS == 0 && D > 0, "D must be a positive multiple of 16");
static_assert(HEAD_NBYTES % HALF_NBYTES == 0, "every head starts at a multiple of HALF_NBYTES");
static_assert(
    HALF_NBYTES % NOC_L1_WRITE_ALIGNMENT_BYTES == 0,
    "output write offsets are multiples of HALF_NBYTES, so it must be a multiple of the L1 NoC write alignment");
static_assert(
    HALF_NBYTES % NOC_DRAM_WRITE_ALIGNMENT_BYTES == 0,
    "output write offsets are multiples of HALF_NBYTES, so it must be a multiple of the DRAM NoC write alignment");
static_assert(!SPLIT_GATHER || value_stick_nbytes == D * 2u, "the split gather packs unpadded D*2-byte sticks");
static_assert(
    !SPLIT_GATHER || sizeof(fused_msda_gather::PointArgs) <= gather_mailbox_nbytes,
    "PointArgs no longer fits the gather mailbox page");

void kernel_main() {
    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t value_addr = get_arg_val<uint32_t>(2);

    const auto output_acc = TensorAccessor(output_args, output_addr, output_page_nbytes);
    [[maybe_unused]] const auto value_acc = TensorAccessor(value_args, value_addr, value_page_nbytes);

    Noc noc;
    CircularBuffer output_tile_cb(output_tile_cb_index);

    [[maybe_unused]] const uint32_t gather_mailbox_l1 =
        SPLIT_GATHER ? CircularBuffer(gather_mailbox_cb_index).get_write_ptr() : 0u;
    [[maybe_unused]] Semaphore<> gather_ready(gather_ready_sem_id);
    [[maybe_unused]] Semaphore<> gather_done(gather_done_sem_id);
    [[maybe_unused]] uint32_t gather_seq = 0;

#ifdef MSDA_DEVICE_COUNTS
    const auto counts_acc = TensorAccessor(counts_args, get_arg_val<uint32_t>(1), MSDA_BATCH * sizeof(uint32_t));
    const auto work = fused_msda_work::stage_work_list<MSDA_BATCH, MSDA_NUM_QUERIES, NUM_HEADS>(
        noc, counts_acc, MSDA_WRITER_COUNTS_CB, get_arg_val<uint32_t>(3), get_arg_val<uint32_t>(4));
    const uint32_t num_output_tiles = work.run.count;
#else
    const uint32_t num_output_tiles = get_arg_val<uint32_t>(1);
    uint32_t arg_idx = 3;
#endif
    for (uint32_t t = 0; t < num_output_tiles; ++t) {
#ifdef MSDA_DEVICE_COUNTS
        const auto u = work.unit(t);
        const uint32_t base_page = u.b * MSDA_NUM_QUERIES + u.q_start;
        const uint32_t head = u.head;
        const uint32_t v_rows = u.v_rows;
#else
        const uint32_t base_page = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t head = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t v_rows = get_arg_val<uint32_t>(arg_idx++);
#endif
        const uint32_t head_byte_offset = head * HEAD_NBYTES;

        if constexpr (SPLIT_GATHER) {
            // The block's output cannot exist before its last point is
            // gathered, so helping with every point first costs no overlap.
            for (uint32_t j = 0; j < POINTS_PER_BLOCK; ++j) {
                // The wait invalidates the RISC L1 data cache on every poll, so
                // the mailbox and the corner tile read after it are the
                // reader's and compute's current writes.
                gather_ready.wait(++gather_seq);
                const auto pt = fused_msda_gather::fetch_point(gather_mailbox_l1);
                fused_msda_gather::gather_rows<WriterGatherCfg>(
                    noc,
                    value_acc,
                    pt,
                    fused_msda_gather::SPLIT_ROW<decltype(value_acc)>,
                    fused_msda_tile_layout::TILE_MAX_ROWS);
                noc.async_read_barrier();
                gather_done.set(gather_seq);
            }
        }

        output_tile_cb.wait_front(N_D_TILES);
        const uint32_t tile_l1 = output_tile_cb.get_read_ptr();

        for (uint32_t r = 0; r < v_rows; ++r) {
            const auto off = fused_msda_tile_layout::tile_row_offsets(r);
            const uint32_t page = base_page + r;
            for (uint32_t k = 0; k < N_D_TILES; ++k) {
                const uint32_t ktile_l1 = tile_l1 + k * TILE_NBYTES;
                const uint32_t dst_offset = head_byte_offset + k * TILE_COLS * sizeof(uint16_t);
                noc.async_write(
                    CoreLocalMem<uint32_t>(ktile_l1 + off.lo),
                    output_acc,
                    HALF_NBYTES,
                    {.offset_bytes = 0},
                    {.page_id = page, .offset_bytes = dst_offset});
                // The last d-tile has no high half when D % 32 == 16.
                if (k * TILE_COLS + HALF_COLS < D) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(ktile_l1 + off.hi),
                        output_acc,
                        HALF_NBYTES,
                        {.offset_bytes = 0},
                        {.page_id = page, .offset_bytes = dst_offset + HALF_NBYTES});
                }
            }
        }
        // The writes read straight from the tile pages, so they must have left
        // before compute may overwrite them.
        noc.async_writes_flushed();
        output_tile_cb.pop_front(N_D_TILES);
    }
    // The writes need only reach the output before the op completes, so one
    // barrier covers every block.
    noc.async_write_barrier();
}

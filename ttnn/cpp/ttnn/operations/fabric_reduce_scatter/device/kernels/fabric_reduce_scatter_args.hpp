// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Circular buffers, kernels and argument slots of the fabric_reduce_scatter kernels. The kernels and the
// program factory index every slot through these names.
//
// Per-core runtime args hold what differs between the cores of one chip (bank sets, schedules, peers);
// common runtime args hold what every core of a kernel gets (tensor addresses, block size, counter
// addresses), so a cache hit writes each address once per kernel.
namespace fabric_reduce_scatter {

// The generic op's ids, unchanged.
namespace cb {
constexpr uint32_t OWN = 0;   // my partial
constexpr uint32_t IN = 1;    // upstream's running sum (port), the forward arrival (final core)
constexpr uint32_t IN2 = 2;   // the backward arrival (final cores only)
constexpr uint32_t OUT = 16;  // the sums, to the sender / final writer
}  // namespace cb

// Kernel order in every chip's ProgramDescriptor::kernels (a kernel's handle in the cached Program).
namespace kernel_index {
enum : uint32_t { PORT_READER, PORT_COMPUTE, PORT_SENDER, FINAL_READER, FINAL_COMPUTE, FINAL_WRITER, COUNT };
}  // namespace kernel_index

// Compile-time args every data-movement kernel starts with. The port sender and the final writer follow
// them with one tensor accessor block (scratch / output) at COUNT; the readers first take the column-block
// geometry (read_ct) and then input's and scratch's accessor blocks.
namespace ct {
enum : uint32_t { PAGE_BYTES, RUN_PAGES, NUM_BANKS, GROUP, INC_EVERY, COUNT };
}  // namespace ct

// Readers only (port reader, final reader), after ct::COUNT: dim=3 column blocks. COL_WB == 0 means row
// blocks (dim=2): block-local page q is input page block * blk_pages + q. Otherwise input page of block j,
// block-local q is (q / COL_WB) * COL_WT + j * COL_WB + q % COL_WB, read page by page.
namespace read_ct {
enum : uint32_t { COL_WT = ct::COUNT, COL_WB, COUNT };
}  // namespace read_ct

// Compute kernels (both variants): tiles per chunk.
namespace compute_ct {
enum : uint32_t { RUN_PAGES, COUNT };
}  // namespace compute_ct

// ---- Port reader -----------------------------------------------------------------------------------
namespace port_reader_common {
enum : uint32_t { IN_ADDR, SCR_ADDR, BLK_PAGES, ARRIVAL_ADDR /* semaphore A */, COUNT };
}  // namespace port_reader_common
// Per core; the block list (one block index per entry, NUM_BLOCKS entries) starts at COUNT.
namespace port_reader_rt {
enum : uint32_t { FIRST_BANK, BANK_STRIDE, FULL, LEAD, NUM_BLOCKS, COUNT };
}  // namespace port_reader_rt

// ---- Compute (ports and final cores) ---------------------------------------------------------------
namespace compute_rt {
enum : uint32_t { N, HAS_B, HAS_C, N_COPY, COUNT };
}  // namespace compute_rt

// ---- Port sender -----------------------------------------------------------------------------------
namespace port_sender_common {
enum : uint32_t { SCR_ADDR, BLK_PAGES, ARRIVAL_ADDR /* semaphore A */, READY_ADDR, COUNT };
}  // namespace port_sender_common
// Per core; then NUM_BLOCKS entries (block | slot << 16) from COUNT, then the fabric connection args
// (tt::tt_fabric::append_fabric_connection_rt_args) when the port sends anything or a ready.
namespace port_sender_rt {
enum : uint32_t {
    FIRST_BANK,
    BANK_STRIDE,
    FINAL_SEM_ADDR,  // semaphore A (forward) or B (backward) on the downstream final cores
    EXPECT_IN,
    FULL,
    SEND_READY,
    READY_X,
    READY_Y,
    PEER_X,
    PEER_Y,
    FINAL_X,
    FINAL_Y,
    FINAL_X1,
    FINAL_Y1,
    FULL_F0,
    FULL_F1,
    DST_MESH_ID,
    DST_CHIP_ID,
    NUM_BLOCKS,
    COUNT
};
}  // namespace port_sender_rt
constexpr uint32_t BLOCK_SLOT_SHIFT = 16;  // sender entry: block | slot << 16

// ---- Final reader ----------------------------------------------------------------------------------
namespace final_reader_common {
enum : uint32_t { IN_ADDR, SCR_ADDR, BLK_PAGES, BWD_SLOT /* G */, SEM_A_ADDR, SEM_B_ADDR, COUNT };
}  // namespace final_reader_common
namespace final_reader_rt {
enum : uint32_t { FIRST_BANK, BANK_STRIDE, OWN_SLOT, FWD_SLOT, HAS_FWD, HAS_BWD, NUM_CHUNKS, COUNT };
}  // namespace final_reader_rt

// ---- Final writer ----------------------------------------------------------------------------------
namespace final_writer_common {
enum : uint32_t { OUT_ADDR, BLK_PAGES, SEM_A_ADDR, SEM_B_ADDR, COUNT };
}  // namespace final_writer_common
namespace final_writer_rt {
enum : uint32_t { FIRST_BANK, BANK_STRIDE, NUM_CHUNKS, EXPECT_A, EXPECT_B, COUNT };
}  // namespace final_writer_rt

}  // namespace fabric_reduce_scatter

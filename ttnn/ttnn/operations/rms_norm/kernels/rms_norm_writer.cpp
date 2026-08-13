// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm writer (BRISC / NoC1). Owns BOTH halves of the cross-core combine,
// the output drain, and every kernel-scope CONSTANT the compute kernel consumes.
//
// Per kernel, once (all three were on the reader's pre-read critical path before;
// this BRISC is idle until the first stat partial exists, so they are free here):
//   prepare_constants()  — cb_wmask (0/1 column mask for a ragged hidden tile),
//                          cb_scaler (the reduce scaler, carrying 1/W_true), and
//                          cb_zero_tile (the combine's identity operand, leaders only).
//
// Per block:
//   combine_stat_block (gather) — every member unicasts its block_row_tiles
//       partial tiles into the ROOT's cb_stat_gather at slot r*G + my_slot, then
//       bumps the gather semaphore. The root waits (block+1)*(G-1) and pushes
//       cb_stat_gather for its compute kernel. The tile-row-major slot layout
//       makes a tile-row's G contributions contiguous, which is what lets the
//       combine be a single eltwise_chain over grid(R, G).
//   combine_stat_block (mcast) — the root multicasts the finalized rstd tiles to
//       the group rectangle with src != dst, so INCLUDE_SRC loopback lands the
//       root's own copy in its cb_rstd too: cb_rstd has exactly one producer
//       (the writer) on EVERY member, root included.
//   store_block — the output block to DRAM, batched one tile-row per barrier.
//
// PLACEMENT. On an INTERLEAVED output, store_block writes DRAM through a
// TensorAccessor. On a physically SHARDED output the block's final home is already
// this core's L1 and there is NO NoC write of it:
//   * TILE shard      — cb_output_tiles is pinned zero-copy over the shard buffer,
//                       so compute packed straight into it and store_block is the
//                       CB handshake alone.
//   * ROW_MAJOR shard — untilize emits the group-uniform tile-row stride, not the
//                       shard's stick stride, so the sticks are re-strided
//                       CORE-LOCALLY (write_shard_rows, L1 -> L1).
// A core's own shard is NEVER addressed through a TensorAccessor.
//
// Helper substitutions (raw NoC instead of a kernel_lib helper), with reasons:
//   * The GATHER leg is raw noc_async_write + semaphore. mcast_pipe's SenderPipe
//     is a one-to-many broadcast of one buffer to a rectangle; the gather is
//     many-to-one into DISJOINT SLOTS of a single destination
//     (mcast_pipe.hpp:44-45 states its precondition as "one sender per receiver,
//     dst_l1 identical on all receivers" — the opposite direction). The RETURN
//     multicast in the same phase does use SenderPipe/ReceiverPipe.
//   * store_block uses raw NoC on BOTH paths. write_sticks_after_untilize is
//     ROW-MAJOR only (tilize_helpers_dataflow.inl:82-85), so it cannot serve the
//     tiled path at all; and on the row-major path it derives BOTH its page
//     count and its L1 row stride from `row_bytes` (inl:203-205,213-216), so it
//     can only drain a block whose row stride equals this core's valid slice.
//     untilize<CB_W> emits the GROUP-UNIFORM stride, which is wider on a ragged
//     core. write_slice_rows() below is the helper's body with the stride taken
//     from the uniform CB width instead of from row_bytes.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

// PER-STAGE INSTRUMENTATION (PERMANENT — see perf_instrumentation.hpp's
// durability contract). The gather and the output drain are each split into
// WAIT / ISSUE / BARRIER: a barrier at ~0 does not mean the transfer was hidden,
// it means the RISC-serial issue cost was paid in the issue zone
// (device-zone-scope-attribution.md §4), and a `*_wait` zone is a rendezvous or
// back-pressure number, never work.

// TEMPORARY ablation switch (Refinement 4 measurement) — MUST be 0 in committed
// code. 1 drops the output DRAM write PAYLOAD while keeping every barrier and CB
// handshake, so the diff against the baseline is the write's contribution.
#define RMSN_ABLATE_OUTPUT_WRITE 0
// TEMPORARY ablation switch (Refinement 5 measurement) — MUST be 0 in committed
// code. 1 shrinks the gather leg's PAYLOAD to a quarter tile (one face) per stat
// tile while keeping the transaction count, the barrier and every semaphore, so the
// diff against the baseline is the gather's NoC-byte contribution. Perf 1 made the
// production payload itself column-valid (see `write_stat_payload`), so this switch
// now ablates on top of that and is expressed in FACES rather than in a byte literal.
#define RMSN_ABLATE_GATHER_BYTES 0
// TEMPORARY ablation switch — MUST be 0 in committed code. 1 removes EVERY cross-core
// dependency: no gather pull, no readiness semaphore, no rstd multicast. Each core just
// hands its own compute the CB quanta it expects (filled with whatever L1 already held),
// so no core ever waits on another. Output is garbage by design. The resulting wall is
// the COMPUTE FLOOR: what the op would cost if every partial were already exactly where
// it needs to be, i.e. the bound on what any data-movement change can possibly buy.
#define RMSN_ABLATE_CROSSCORE 0

namespace {
constexpr uint32_t cb_scaler = 2;
constexpr uint32_t cb_wmask = 3;
constexpr uint32_t cb_zero_tile = 4;
constexpr uint32_t cb_stat_partial = 7;
constexpr uint32_t cb_stat_gather = 8;
constexpr uint32_t cb_stat_gather2 = 15;  // two-stage combine only (root)
constexpr uint32_t cb_branch_sum = 18;    // two-stage combine only (row leaders)
constexpr uint32_t cb_rstd_send = 10;
constexpr uint32_t cb_rstd = 11;
constexpr uint32_t cb_output_tiles = 16;
constexpr uint32_t cb_output_rm = 17;
constexpr uint32_t TILE_HW_DIM = 32;

// ---- COLUMN-VALID CROSS-CORE STAT PAYLOAD (Perf 1, idea I2) ----------------
//
// Every tile that crosses the NoC in the combine is a `reduce<SUM, REDUCE_ROW>`
// result, i.e. COLUMN-0-VALID: only column 0 of each of the 32 rows carries
// information. In a tile's face order (four 16x16 faces, row-major within a face)
// column 0 lives in FACE 0 (rows 0..15) and FACE 2 (rows 16..31) — byte ranges
// [0, T/4) and [T/2, 3T/4). So half of every stat tile need never be sent.
//
// Faces 1 and 3 of a destination slot are then never written and hold arbitrary
// pre-existing L1. That is safe, and it was PROVEN rather than argued: with bf16 NaN
// stamped into face 1 and -Inf into face 3 of every gather slot and of the rstd
// landing tile, PCC is UNCHANGED on decode7168 / wshard1024 / bshard1024. Three
// independent reasons — the combine's FPU Add is elementwise (garbage cannot migrate
// into column 0), the finalize SFPU runs `VectorMode::C` (faces 0+2 only), and the
// apply consumes rstd as `OperandKind::Col` (column 0 alone). No zero-fill is needed;
// one was priced anyway at +20.3 us (decode7168) / +139 us (bshard1024), an order of
// magnitude more than the win, so if it WERE needed this idea would be dead.
//
// MEASURED (blackhole_p150b, bf16 / TILE / HiFi2 / fp32_dest_acc_en=False, median n=4):
//   (1,1,8192,1024) BLOCK_SHARDED [1024,128] (8,8)  50453 -> 47935 ns  (-5.0 %)
//   (1,1,32,1024)   WIDTH_SHARDED [32,128]  (8,1)    4564 ->  4421 ns  (-3.1 %)
//   (1,1,32,7168)   WIDTH_SHARDED [32,256]  (7,4)    5844 ->  5698 ns  (-2.5 %)
//   (1,1,32,2047)   interleaved, ragged hidden       6506 ->  6313 ns  (-3.0 %)
//   interleaved decode/prefill, HEIGHT (G=1)         flat, inside the noise band
// The whole-op delta on the BLOCK geometry is the gather rendezvous almost 1:1
// (`wr_gather_issue` 6291 -> 3839, `wr_gather_sem_wait` 5367 -> 2655, `cp_gather_wait`
// 7270 -> 4684 ns). This leg is destination-L1/NoC-BYTE bound, not issue bound:
// halving the bytes while DOUBLING the transactions beats the 1-transaction
// three-quarter-tile encoding (which measured only -2.2 % on the same geometry).
//
// RAW-NoC JUSTIFICATION: the gather leg was already raw `noc_async_write` (see the
// header note — mcast_pipe's SenderPipe is one-to-many with an identical dst, the
// gather is many-to-one into disjoint slots). This only changes its byte count, plus
// the split into the two face-sized transfers that ONE contiguous transfer cannot
// express (faces 0 and 2 are T/2 apart). The MULTICAST stays on the helper:
// `SenderPipe::send(src, dst, size)` takes one contiguous byte count, so a
// strided/multi-chunk broadcast is inexpressible with it — there, the lever is only
// the trailing-garbage trim (drop the LAST tile's face 3), which it expresses natively.
FORCE_INLINE void write_stat_payload(uint32_t src, uint64_t dst, uint32_t tile_bytes) {
    const uint32_t face = tile_bytes >> 2;
#if RMSN_ABLATE_GATHER_BYTES
    noc_async_write(src, dst, face);  // ablation: face 0 only (numerically wrong)
#else
    noc_async_write(src, dst, face);                        // face 0 == rows 0..15, col 0
    noc_async_write(src + 2 * face, dst + 2 * face, face);  // face 2 == rows 16..31, col 0
#endif
}

// A resident-shard "accessor" with the same page/offset shape as TensorAccessor,
// resolving to THIS core's own L1 (the mirror of the reader's). Lets
// write_slice_rows serve the DRAM leg and the resident-shard leg with one body.
struct LocalShardAccessor {
    uint32_t base;
    uint32_t row_bytes;
    FORCE_INLINE uint64_t get_noc_addr(uint32_t page, uint32_t byte_offset) const {
        return ::get_noc_addr(base + page * row_bytes + byte_offset);
    }
};

// Drain one uniform-width CB block (`cb_w_tiles` tile-sized pages) as `rows`
// row-major sticks of `slice_bytes`, reading each stick at the block's uniform
// L1 row stride. The trailing pad columns are never written out.
template <uint32_t cb_id, uint32_t cb_w_tiles, typename Accessor>
FORCE_INLINE void write_slice_rows(
    const Accessor& acc, uint32_t rows, uint32_t slice_bytes, uint32_t start_page, uint32_t byte_offset) {
    constexpr uint32_t tile_row_bytes = get_tile_size(cb_id) / TILE_HW_DIM;
    constexpr uint32_t block_row_bytes = tile_row_bytes * cb_w_tiles;
    cb_wait_front(cb_id, cb_w_tiles);
    uint32_t l1_addr = get_read_ptr(cb_id);
    for (uint32_t r = 0; r < rows; ++r) {
        // `slice_bytes == 0`: a hidden chunk entirely past this core's valid
        // slice. Its pages are still consumed (the quantum is uniform), just
        // never stored.
        if (slice_bytes) {
            noc_async_write(l1_addr, acc.get_noc_addr(start_page + r, byte_offset), slice_bytes);
        }
        l1_addr += block_row_bytes;
    }
    // One barrier per 32-row block == cb_w_tiles tiles per barrier.
    noc_async_write_barrier();
    cb_pop_front(cb_id, cb_w_tiles);
}

// The resident-shard mirror of read_shard_rows: a core-local L1 re-stride of the
// untilized block back into this core's OWN output shard. No DRAM crossing. When
// the shard's stick stride already IS the block row stride the whole 32-row group
// goes out as ONE transfer instead of 32.
template <uint32_t cb_id, uint32_t cb_w_tiles>
FORCE_INLINE void write_shard_rows(
    uint32_t base,
    uint32_t shard_row_bytes,
    uint32_t rows,
    uint32_t slice_bytes,
    uint32_t start_page,
    uint32_t byte_offset) {
    constexpr uint32_t block_row_bytes = (get_tile_size(cb_id) / TILE_HW_DIM) * cb_w_tiles;
    if (slice_bytes == block_row_bytes && shard_row_bytes == block_row_bytes) {
        cb_wait_front(cb_id, cb_w_tiles);
        noc_async_write(
            get_read_ptr(cb_id),
            ::get_noc_addr(base + start_page * shard_row_bytes + byte_offset),
            rows * block_row_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id, cb_w_tiles);
        return;
    }
    write_slice_rows<cb_id, cb_w_tiles>(
        LocalShardAccessor{base, shard_row_bytes}, rows, slice_bytes, start_page, byte_offset);
}
}  // namespace

using namespace dataflow_kernel_lib;

void kernel_main() {
    constexpr uint32_t CB_W_TILES = get_compile_time_arg_val(0);
    constexpr uint32_t TENSOR_W_TILES = get_compile_time_arg_val(1);
    constexpr bool IS_RM_OUT = get_compile_time_arg_val(2) != 0;
    constexpr uint32_t W_GROUP_SIZE = get_compile_time_arg_val(3);
    constexpr uint32_t SEM_GATHER = get_compile_time_arg_val(4);
    constexpr bool IS_SHARDED_OUT = get_compile_time_arg_val(5) != 0;
    // Block extent along `hidden` of the streaming CBs (cb_output_rm, and
    // cb_output_tiles on the interleaved leg). Equal to CB_W_TILES — one chunk,
    // the unchunked schedule — unless a resident shard pinned a hidden slice too
    // wide to hold. See the compute kernel's "HIDDEN-AXIS CHUNKING" note.
    constexpr uint32_t CB_CHUNK_TILES = get_compile_time_arg_val(6);
    constexpr uint32_t NUM_CHUNKS = (CB_W_TILES + CB_CHUNK_TILES - 1) / CB_CHUNK_TILES;
    // Level-2 fan-in of the combine tree. 1 == the flat root-gather (every member
    // unicasts straight to the root and W_GROUP_SIZE above is the whole group), in
    // which case every `if constexpr (TWO_STAGE)` below compiles out and this
    // kernel is byte-identical to the Phase 0 one.
    constexpr uint32_t STAGE2_SPAN = get_compile_time_arg_val(7);
    constexpr uint32_t SEM_GATHER2 = get_compile_time_arg_val(8);
    // Perf 1 (I4): this kernel prepares the reduce scaler and the ragged-hidden mask,
    // so it carries the reader's old HAS_ANY_TAIL gate.
    constexpr bool HAS_ANY_TAIL = get_compile_time_arg_val(9) != 0;
    constexpr bool TWO_STAGE = STAGE2_SPAN > 1;
    // FLAT combine: no tree, and (since the row-split) no root either. Every core is
    // a WORKER owning a slice of the tile-rows, and the finalized rstd is all-gathered
    // rather than multicast from one core.
    constexpr bool FLAT = !TWO_STAGE;
    constexpr uint32_t MCAST_CT_BASE = 10;
    constexpr uint32_t MCAST_RT_BASE = 22;
    constexpr auto mc = McastArgs<MCAST_CT_BASE, MCAST_RT_BASE>();
    constexpr auto dst_args = TensorAccessorArgs<mc.next_compile_time_args_offset()>();
    // The pull gather needs EVERY group member's coordinates, not just the leader's:
    // a group's physical columns are not contiguous (worker columns skip the DRAM /
    // PCIe / ARC columns — a (8,1) group lands on x = 1..6, 11, 12), so the root
    // cannot derive peer coords by arithmetic. The host emits a flat
    // [x0, y0, x1, y1, ...] table AFTER the multicast runtime args, whose length the
    // McastArgs block reports, so the existing MCAST_RT_BASE contract is untouched.
    constexpr uint32_t MEMBER_RT_BASE = mc.next_runtime_args_offset();

    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t row_tile_start = get_arg_val<uint32_t>(1);
    const uint32_t num_blocks = get_arg_val<uint32_t>(2);
    const uint32_t block_row_tiles = get_arg_val<uint32_t>(3);
    const uint32_t last_block_row_tiles = get_arg_val<uint32_t>(4);
    const uint32_t w_tile_start = get_arg_val<uint32_t>(5);
    const uint32_t core_w = get_arg_val<uint32_t>(6);
    const uint32_t my_slot = get_arg_val<uint32_t>(7);  // level-1 slot in the leader's gather
    const uint32_t is_root = get_arg_val<uint32_t>(8);
    const uint32_t root_x = get_arg_val<uint32_t>(9);
    const uint32_t root_y = get_arg_val<uint32_t>(10);
    const uint32_t num_sticks = get_arg_val<uint32_t>(11);
    const uint32_t stick_start = get_arg_val<uint32_t>(12);
    const uint32_t out_slice_bytes = get_arg_val<uint32_t>(13);
    const uint32_t out_byte_offset = get_arg_val<uint32_t>(14);
    [[maybe_unused]] const uint32_t shard_row_bytes = get_arg_val<uint32_t>(15);
    // Combine tree, level 1: this core's destination is its grid ROW's leader. Flat
    // (STAGE2_SPAN == 1) sets leader == root and is_leader == is_root on the host,
    // so the gather leg below is unchanged there.
    const uint32_t leader_x = get_arg_val<uint32_t>(16);
    const uint32_t leader_y = get_arg_val<uint32_t>(17);
    const uint32_t is_leader = get_arg_val<uint32_t>(18);
    [[maybe_unused]] const uint32_t my_row_slot = get_arg_val<uint32_t>(19);  // level-2 slot on the root
    // Perf 1 (I4): the two reduce constants, moved here from the reader.
    const uint32_t core_partial_w = get_arg_val<uint32_t>(20);
    const uint32_t inv_w_bits = get_arg_val<uint32_t>(21);

    // A mcast-box FILLER core (see the reader): inside a group's broadcast rectangle
    // but owning no shard. It carries no work, never gathers and never acks — the
    // mcast is emitted with an explicit num_active = member count for exactly this.
    if (num_blocks == 0) {
        return;
    }

    Noc noc;
    Semaphore<> gather_sem(SEM_GATHER);
    [[maybe_unused]] Semaphore<> gather2_sem(SEM_GATHER2);

    // ---- prepare_constants (cb_wmask, cb_scaler) — MOVED FROM THE READER --------
    // Perf 1, idea I4. These two helper calls cost ~310 ns EACH and used to sit in
    // the reader AHEAD of the input block read, i.e. on the critical path of every
    // core, for buffers whose consumers cannot run until the whole block has landed.
    // This BRISC is idle until this core's first stat partial exists (measured
    // `wr_partial_wait` p50 >= 1.1 us on every geometry), so the fill is free here.
    // MEASURED (110 cores, blackhole_p150b, bf16/HiFi2/fp32_dest_acc_en=False):
    // C=1 2476 -> 2221, C=2 3007 -> 2779 (1.08x), C=3 3425 -> 3213,
    // C=3 with a ragged tail 3977 -> 3353 (1.19x), 8 cores C=2 1953 -> 1630 ns.
    // MASK FIRST, and both BEFORE the cb_zero_tile fill below: the mask's consumer
    // (mask_tail_block, inside sumsq) has the earliest deadline of the three, the
    // scaler's (the reduce) is next, and cb_zero_tile is not read until the combine.
    // Producer is still exactly one kernel per CB, just a different one.
    if constexpr (HAS_ANY_TAIL) {
        if (core_partial_w != 0) {
            MaybeDeviceZoneScope("wr_prep_mask");
            // 1.0 in columns [0, core_partial_w), 0 elsewhere, in the row-0 broadcast
            // layout the compute kernel consumes with BroadcastDim::Row. PER-CORE, not
            // per-tensor: a ROW_MAJOR WIDTH/BLOCK shard's width granule is the L1
            // alignment, so EVERY core's hidden slice can end mid-tile.
            dataflow_kernel_lib::prepare_reduce_mask<cb_wmask, ckernel::ReduceDim::REDUCE_ROW>(core_partial_w);
        }
    }
    {
        // Pool-type-aware overload: SUM/REDUCE_ROW fills the matmul-path scaler
        // layout. The scaler is 1/W_true, not 1.0 (Refinement 5) — the NON-STANDARD
        // scaler case the reduce helpers' own header names ("the scaler combines
        // reduction with another factor"), which is what lets the divisor ride a
        // multiply the reduce performs anyway instead of an SFPU pass on the root.
        MaybeDeviceZoneScope("wr_prep_scaler");
        float inv_w;
        __builtin_memcpy(&inv_w, &inv_w_bits, sizeof(inv_w));
        dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>(
            inv_w);
    }

    // ---- prepare_constants (cb_zero_tile), ROOT ONLY ------------------------
    // The identity B operand of the combine's Add accumulation (BinaryFpu needs two
    // CB inputs), read by the compute kernel of every core that runs a combine
    // chain — the row LEADERS (which is the root alone when the tree is flat, since
    // is_leader == is_root there) and nobody else.
    //
    // ONLY ODD GROUP SIZES STILL NEED IT. The combine chains sum their gathered
    // tiles PAIRWISE (`gather[c] + gather[c + span/2]`, accumulated in DEST), which
    // needs no identity operand at all — see `cp_combine_l1` in the compute kernel.
    // The halves only tile when the span is even, so an odd span falls back to the
    // zero-padded chain and is the sole remaining consumer of this tile.
    //
    // NEVER FILL A CONSTANT TILE WITH A CPU STORE LOOP. This used to be a
    // 4096-byte scalar dword loop, justified by "this BRISC is idle until the
    // first partial anyway, so the fill is free". That premise is geometry-
    // dependent and FALSE on the small sharded geometries: at WIDTH [32,128] x
    // (8,1) the root's own partial is ready in ~260 ns (`wr_partial_wait` p50),
    // so a 1211 ns fill sat squarely on the root's serial path and was, on its
    // own, larger than this geometry's entire gap to production rms_norm.
    // `noc.async_write_zeros` does the same fill as a chunked NoC loopback read
    // from MEM_ZEROS_BASE (1211 -> 217 ns); the pairwise combine then removed the
    // remaining 217 for every even span.
    constexpr bool NEEDS_ZERO_TILE = (W_GROUP_SIZE % 2 != 0) || (TWO_STAGE && STAGE2_SPAN % 2 != 0);
    // WHO needs it changed with the row-split: the FLAT combine now runs on EVERY core
    // (each folds its own slice of tile-rows), not just on a single leader, so every
    // core must have the identity operand. Only the two-stage tree still concentrates
    // the fold on its leaders.
    if (NEEDS_ZERO_TILE && (FLAT || is_leader)) {
        MaybeDeviceZoneScope("wr_zero_fill");
        CircularBuffer cb_zero_tile_obj(cb_zero_tile);
        cb_zero_tile_obj.reserve_back(1);
        noc.async_write_zeros(cb_zero_tile_obj, get_tile_size(cb_zero_tile), {.offset_bytes = 0});
        noc.write_zeros_l1_barrier();
        cb_zero_tile_obj.push_back(1);
    }

    const uint32_t stat_tile_bytes = get_tile_size(cb_stat_gather);
    // Is the row-split worth taking? It parallelises combine+finalize across the group,
    // but only if there is more than one tile-row's worth of work to hand out. At
    // rows_t == 1 (the decode-shaped width shards) there is exactly ONE worker, so the
    // split buys nothing and would trade an efficient one-to-many multicast for G
    // unicasts from a single core — measured 3526 -> 4233 ns on (1,1,32,1024). Keep the
    // proven root-gather + multicast there and split only when the work divides.
    // MIRRORS the host's `_row_split_applies`: derived from PROGRAM constants only.
    // Deriving it from block_row_tiles would let host and device disagree whenever the
    // residency solve lands on R == 1, and the host sizes cb_stat_gather on this very
    // predicate (streaming window when split, full R*G landing zone when not).
    const uint32_t core_row_tiles = (num_blocks == 0) ? 0 : ((num_blocks - 1) * block_row_tiles + last_block_row_tiles);
    const bool SPLIT = FLAT && (W_GROUP_SIZE > 1) && (core_row_tiles > 1);
    // cb_stat_gather holds exactly block_row_tiles * W_GROUP_SIZE pages and the
    // leader pushes/pops that many per block, so its write pointer is back at the
    // CB base at the start of every block — identical on every group member,
    // which is what lets a member address the leader's slots by local pointer.
    const uint32_t gather_base = get_write_ptr(cb_stat_gather);
    // Same argument, one level up: block_row_tiles * STAGE2_SPAN pages, pushed and
    // popped whole every block, so the base is stable and group-uniform.
    [[maybe_unused]] const uint32_t gather2_base = TWO_STAGE ? get_write_ptr(cb_stat_gather2) : 0;

    const uint32_t out_tile_bytes = get_tile_size(cb_output_tiles);
    // Default page size == the accessor args' aligned page size, which is the
    // tile size on the tiled path and the stick size on the row-major path.
    // Built unconditionally (one type for every leg, so `if constexpr` in the
    // non-template store_block below still type-checks), but a SHARDED output never
    // USES it: its shard is this core's own L1, addressed through
    // LocalShardAccessor. The accessor keeps owning interleaved I/O only.
    [[maybe_unused]] const auto out_acc = TensorAccessor(dst_args, dst_addr);

    // --- store_block: drain one output block to DRAM -------------------------
    // The hidden axis is drained in NUM_CHUNKS chunks of CB_CHUNK_TILES, in the
    // order compute produces them; at NUM_CHUNKS == 1 every loop runs once and this
    // is the Phase 0 schedule. A TILE output needs no chunking — it is either the
    // pinned shard (compute packed straight into it) or written from a whole-block
    // CB the interleaved leg never chunks.
    uint32_t sticks_done = 0;
    auto store_block = [&](uint32_t b, uint32_t rows_t) {
        if constexpr (IS_RM_OUT) {
            MaybeDeviceZoneScope("wr_out_rm_restride");
            constexpr uint32_t out_chunk_bytes = (get_tile_size(cb_output_rm) / TILE_HW_DIM) * CB_CHUNK_TILES;
            // ROW_MAJOR output, resident shard or DRAM. untilize emits the
            // group-uniform tile-row stride, which is neither the shard's stick
            // stride nor the DRAM row, so both legs re-stride the sticks — the
            // sharded one core-locally (L1 -> L1, no DRAM crossing).
            for (uint32_t k = 0; k < NUM_CHUNKS; ++k) {
                const uint32_t chunk_off = k * out_chunk_bytes;
                const uint32_t chunk_bytes =
                    (out_slice_bytes > chunk_off)
                        ? ((out_slice_bytes - chunk_off < out_chunk_bytes) ? out_slice_bytes - chunk_off
                                                                           : out_chunk_bytes)
                        : 0;
                uint32_t done = sticks_done;
                for (uint32_t r = 0; r < rows_t; ++r) {
                    uint32_t sticks_this = TILE_HW_DIM;
                    if (sticks_this > num_sticks - done) {
                        sticks_this = num_sticks - done;
                    }
                    if constexpr (IS_SHARDED_OUT) {
                        write_shard_rows<cb_output_rm, CB_CHUNK_TILES>(
                            dst_addr, shard_row_bytes, sticks_this, chunk_bytes, stick_start + done, chunk_off);
                    } else {
                        write_slice_rows<cb_output_rm, CB_CHUNK_TILES>(
                            out_acc, sticks_this, chunk_bytes, stick_start + done, out_byte_offset + chunk_off);
                    }
                    done += sticks_this;
                }
            }
            // Every chunk re-walked the same 32-row groups; advance past them once.
            const uint32_t block_sticks = rows_t * TILE_HW_DIM;
            sticks_done += (block_sticks > num_sticks - sticks_done) ? (num_sticks - sticks_done) : block_sticks;
        } else if constexpr (IS_SHARDED_OUT) {
            // Resident TILE output shard: cb_output_tiles is PINNED zero-copy over
            // it, so compute packed the block straight into its final home and
            // "store_block" is the CB handshake alone — zero NoC traffic. The pad
            // columns / padded tile-rows compute also wrote land in the shard's own
            // padding, which is outside the logical tensor and never read back.
            MaybeDeviceZoneScope("wr_out_pin_wait");
            cb_wait_front(cb_output_tiles, rows_t * CB_W_TILES);
            cb_pop_front(cb_output_tiles, rows_t * CB_W_TILES);
        } else {
            for (uint32_t r = 0; r < rows_t; ++r) {
                {
                    // Starved on the APPLY pass, not on the NoC.
                    MaybeDeviceZoneScope("wr_out_wait");
                    cb_wait_front(cb_output_tiles, CB_W_TILES);
                }
                const uint32_t src = get_read_ptr(cb_output_tiles);
                const uint32_t row_tile = row_tile_start + b * block_row_tiles + r;
                const uint32_t base = row_tile * TENSOR_W_TILES + w_tile_start;
                {
                    // RISC-serial issue; scales with core_w transactions.
                    MaybeDeviceZoneScope("wr_out_issue");
#if !RMSN_ABLATE_OUTPUT_WRITE
                    for (uint32_t c = 0; c < core_w; ++c) {
                        noc_async_write_tile(base + c, out_acc, src + c * out_tile_bytes);
                    }
#endif
                }
                {
                    // One barrier per tile-row (core_w tiles), never per tile.
                    MaybeDeviceZoneScope("wr_out_barrier");
                    noc_async_write_barrier();
                }
                cb_pop_front(cb_output_tiles, CB_W_TILES);
            }
        }
    };

    // --- combine_stat_block (gather leg) -------------------------------------
    //
    // WHY A MEMBER'S WRITE CANNOT RACE THE ROOT'S PREVIOUS BLOCK. A member writes
    // straight into the root's cb_stat_gather slots without inspecting the root's
    // CB space, so the only thing keeping block b+1's partials off block b's
    // still-unread data is the ORDERING the multicast imposes:
    //   member: gather(b) -> receive(b) -> ... -> gather(b+1)
    //   root:   gather(b) -> [compute pops cb_stat_gather in the combine chain,
    //           then pushes cb_rstd_send in the finalize chain] -> send(b)
    // receive(b) cannot return before send(b), and send(b) cannot start before
    // cb_rstd_send holds block b — which the finalize chain only produces AFTER
    // the combine chain has consumed the gather buffer. So the mcast doubles as
    // the group-wide barrier that frees the slots. Any restructuring that lets a
    // member start block b+1 without having received block b (e.g. a deeper
    // cb_rstd, or dropping the pre-handshake) must add explicit back-pressure on
    // cb_stat_gather.
    // Level 1: every member -> its row LEADER's cb_stat_gather, slot r*S1 + my_slot.
    // Flat (STAGE2_SPAN == 1) has leader == root and W_GROUP_SIZE == G, i.e. the
    // Phase 0 gather verbatim.
    // PULL, NOT PUSH (flat combine). A member no longer writes its partial anywhere:
    // it announces readiness with a semaphore and leaves the tiles in its OWN
    // cb_stat_partial, where they stay valid until the block's multicast lands. The
    // root then READS row r's G partials into a GATHER_DEPTH*G streaming window and
    // its compute pops them before the root issues row r+1.
    //
    // Why this is the whole point: with push, the G members write whenever they are
    // ready and never inspect the root's CB, so the root needs a landing slot for
    // every (row, member) pair — O(R*G) resident, which is what forced
    // MAX_GATHER_TILES to cut a tall shard into several blocks. With pull the
    // CONSUMER initiates, so back-pressure is implicit (it simply does not issue the
    // next row until it has space) and the buffer is O(G) for any R. The R-scaling
    // moves to each member's own cb_stat_partial, where it was already paid.
    //
    // The payload is a whole stat tile rather than the column-valid prefix the push
    // path sent: the reader picks the byte count here and the extra bytes are free —
    // tt-npe puts this op at ~1% average / 9.6% peak NoC link utilisation with 0.0%
    // congestion, so the gather is nowhere near bandwidth-bound.
    auto gather_partials = [&](uint32_t b, uint32_t rows_t) {
        {
            // Starved on this core's OWN statistics pipeline (reader read ->
            // sumsq -> reduce). On a member this is the whole pre-combine chain.
            MaybeDeviceZoneScope("wr_partial_wait");
            cb_wait_front(cb_stat_partial, rows_t);
        }
#if RMSN_ABLATE_CROSSCORE
        // No readiness signal, no pull: the leader simply publishes the CB quanta its
        // compute expects and every core frees its own partials immediately.
        if (is_leader) {
            for (uint32_t r = 0; r < rows_t; ++r) {
                cb_reserve_back(cb_stat_gather, W_GROUP_SIZE);
                cb_push_back(cb_stat_gather, W_GROUP_SIZE);
            }
        }
        cb_pop_front(cb_stat_partial, rows_t);
        return;
#endif
        if (SPLIT) {
            // SYMMETRIC READINESS. Every core tells every PEER "my partials exist";
            // there is no root in the flat combine any more. G-1 atomics per core,
            // all issued in parallel, and no data moves.
            if constexpr (W_GROUP_SIZE > 1) {
                for (uint32_t m = 0; m < W_GROUP_SIZE; ++m) {
                    if (m == my_slot) {
                        continue;
                    }
                    gather_sem.up(
                        noc,
                        get_arg_val<uint32_t>(MEMBER_RT_BASE + 2 * m),
                        get_arg_val<uint32_t>(MEMBER_RT_BASE + 2 * m + 1),
                        1);
                }
            }
            {
                MaybeDeviceZoneScope("wr_gather_sem_wait");
                if constexpr (W_GROUP_SIZE > 1) {
                    gather_sem.wait_min((b + 1) * (W_GROUP_SIZE - 1));
                }
            }
            // Pull ONLY this worker's slice of the tile-rows.
            const uint32_t rpw = (rows_t + W_GROUP_SIZE - 1) / W_GROUP_SIZE;
            const uint32_t first = my_slot * rpw;
            const uint32_t mine = (first >= rows_t) ? 0 : ((rows_t - first < rpw) ? (rows_t - first) : rpw);
            const uint32_t peer_partial = get_read_ptr(cb_stat_partial);
            // NOTE: not "wr_gather_issue" — that name at this line hashes to the same
            // 16-bit zone id as the TRISC-FW firmware zone, which the profiler rejects
            // at read time ("Source location hashes are colliding").
            MaybeDeviceZoneScope("wr_gather_pull");
            for (uint32_t r = first; r < first + mine; ++r) {
                cb_reserve_back(cb_stat_gather, W_GROUP_SIZE);
                const uint32_t dst = get_write_ptr(cb_stat_gather);
                const uint32_t src = peer_partial + r * stat_tile_bytes;
                for (uint32_t m = 0; m < W_GROUP_SIZE; ++m) {
                    const uint32_t mx = get_arg_val<uint32_t>(MEMBER_RT_BASE + 2 * m);
                    const uint32_t my = get_arg_val<uint32_t>(MEMBER_RT_BASE + 2 * m + 1);
                    noc_async_read(get_noc_addr(mx, my, src), dst + m * stat_tile_bytes, stat_tile_bytes);
                }
                noc_async_read_barrier();
                cb_push_back(cb_stat_gather, W_GROUP_SIZE);
            }
            return;
        }
        // NON-SPLIT (rows_t == 1): the original PUSH gather. Every member writes its
        // single partial into the leader's slot in parallel and bumps the readiness
        // semaphore. Pull is deliberately NOT used here: with one tile-row there is one
        // worker, so a pull would serialise G read-issues onto that one core for no
        // parallelism in return (measured 3526 -> 3729 ns on (1,1,32,1024)).
        const uint32_t src = get_read_ptr(cb_stat_partial);
        {
            MaybeDeviceZoneScope("wr_gather_push");
            for (uint32_t r = 0; r < rows_t; ++r) {
                const uint32_t dst = gather_base + (r * W_GROUP_SIZE + my_slot) * stat_tile_bytes;
                write_stat_payload(src + r * stat_tile_bytes, get_noc_addr(leader_x, leader_y, dst), stat_tile_bytes);
            }
        }
        {
            MaybeDeviceZoneScope("wr_gather_barrier");
            noc_async_write_barrier();
        }
        cb_pop_front(cb_stat_partial, rows_t);
        if constexpr (W_GROUP_SIZE > 1) {
            if (!is_leader) {
                // Ordered behind the write barrier, so the partial has landed before
                // the leader can observe the count.
                gather_sem.up(noc, leader_x, leader_y, 1);
            }
        }
        if (is_leader) {
            MaybeDeviceZoneScope("wr_gather_sem_wait");
            cb_reserve_back(cb_stat_gather, rows_t * W_GROUP_SIZE);
            if constexpr (W_GROUP_SIZE > 1) {
                gather_sem.wait_min((b + 1) * (W_GROUP_SIZE - 1));
            }
            cb_push_back(cb_stat_gather, rows_t * W_GROUP_SIZE);
        }
    };

    // --- distribute_rstd (FLAT + row-split) ----------------------------------
    //
    // COLLECT-THEN-BROADCAST, the shape production layernorm uses. Each worker
    // finalized its OWN slice of the tile-rows, so the finished rstd starts scattered
    // across the group and every core needs all of it.
    //
    // Every worker writes its slice into the ROOT's cb_rstd at its row offset (the
    // root writes its own locally), signals, and the root then multicasts the WHOLE
    // block once. That is `G-1` small writes plus ONE multicast, against the
    // symmetric all-gather's G writes PER WORKER (64 transactions at G=8).
    //
    // It also keeps a SINGLE multicast sender, which is what makes the multicast
    // usable at all here: G concurrent loopback multicasts over one rectangle do
    // complete (once the rectangle is given start=HIGH, end=LOW for NoC1) but they
    // contend on path reservation — measured median 23053 ns vs 22164 for unicast,
    // with 11x the run-to-run spread. One sender has no such contention, and it is
    // the mcast_pipe precondition ("single sender per receiver") rather than a
    // violation of it.
    //
    // src == dst on the send: cb_rstd is both the root's assembled copy and the
    // landing buffer, so the multicast EXCLUDES self (the root already holds it),
    // which is exactly the src != dst / INCLUDE_SRC rule read the other way.
    [[maybe_unused]] uint32_t rstd_expected = 0;
    auto rstd_slice = [&](uint32_t rows_t, uint32_t& first, uint32_t& mine, uint32_t& workers) {
        const uint32_t rpw = (rows_t + W_GROUP_SIZE - 1) / W_GROUP_SIZE;
        first = my_slot * rpw;
        mine = (first >= rows_t) ? 0 : ((rows_t - first < rpw) ? (rows_t - first) : rpw);
        workers = (rows_t + rpw - 1) / rpw;
    };

    // Place this core's finished slice into the ROOT's cb_rstd, then release it.
    auto contribute_rstd = [&](uint32_t rows_t, uint32_t rstd_base) {
        uint32_t first, mine, workers;
        rstd_slice(rows_t, first, mine, workers);
        if (mine == 0) {
            return workers;
        }
        {
            // Starved on THIS core's own combine + finalize — now 1/G of the work the
            // single root used to do alone.
            MaybeDeviceZoneScope("wr_rstd_own_wait");
            cb_wait_front(cb_rstd_send, mine);
        }
        {
            MaybeDeviceZoneScope("wr_rstd_push");
            noc_async_write(
                get_read_ptr(cb_rstd_send),
                get_noc_addr(leader_x, leader_y, rstd_base + first * stat_tile_bytes),
                mine * stat_tile_bytes);
            noc_async_write_barrier();
        }
        if (!is_leader) {
            gather2_sem.up(noc, leader_x, leader_y, 1);
        }
        cb_pop_front(cb_rstd_send, mine);
        return workers;
    };

    // Level 2 (two-stage only): every row LEADER -> the root's cb_stat_gather2,
    // slot r*S2 + my_row_slot. Compiled out entirely when the combine is flat.
    // The race argument of the level-1 gather carries over unchanged: a leader
    // cannot reach block b+1's level-2 write without having RECEIVED the block-b
    // multicast, which the root only sends after its compute has popped
    // cb_stat_gather2 for block b.
    [[maybe_unused]] auto gather_rows = [&](uint32_t b, uint32_t rows_t) {
        if constexpr (TWO_STAGE) {
            if (!is_leader) {
                return;
            }
            {
                MaybeDeviceZoneScope("wr_branch_wait");
                cb_wait_front(cb_branch_sum, rows_t);
            }
            const uint32_t src = get_read_ptr(cb_branch_sum);
            {
                MaybeDeviceZoneScope("wr_gather2_issue");
                for (uint32_t r = 0; r < rows_t; ++r) {
                    const uint32_t dst = gather2_base + (r * STAGE2_SPAN + my_row_slot) * stat_tile_bytes;
                    write_stat_payload(src + r * stat_tile_bytes, get_noc_addr(root_x, root_y, dst), stat_tile_bytes);
                }
                noc_async_write_barrier();
            }
            cb_pop_front(cb_branch_sum, rows_t);
            if (!is_root) {
                gather2_sem.up(noc, root_x, root_y, 1);
            }
            if (is_root) {
                MaybeDeviceZoneScope("wr_gather2_sem_wait");
                cb_reserve_back(cb_stat_gather2, rows_t * STAGE2_SPAN);
                gather2_sem.wait_min((b + 1) * (STAGE2_SPAN - 1));
                cb_push_back(cb_stat_gather2, rows_t * STAGE2_SPAN);
            }
        }
    };

    if (SPLIT) {
        // Row-split: every core pulls and finalizes its OWN slice, contributes it to
        // the root, and the root broadcasts the assembled block once. The mcast faces
        // are built ONCE outside the loop (the ReceiverPipe ctor kernel-inits its
        // data_ready cell, which must not re-run after the sender has started).
        if (is_leader) {
            auto sender = mc.sender(noc);
            for (uint32_t b = 0; b < num_blocks; ++b) {
                const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
                gather_partials(b, rows_t);
                cb_reserve_back(cb_rstd, rows_t);
                const uint32_t rstd_base = get_write_ptr(cb_rstd);
                const uint32_t workers = contribute_rstd(rows_t, rstd_base);
                {
                    // Waiting on the OTHER workers' slices — 1/G of a combine each,
                    // running concurrently, where this used to be the root's whole
                    // serial combine + finalize.
                    MaybeDeviceZoneScope("wr_rstd_peer_wait");
                    if (workers > 1) {
                        rstd_expected += workers - 1;
                        gather2_sem.wait_min(rstd_expected);
                    }
                }
                {
                    // src == dst: the root already holds the assembled block, so the
                    // multicast excludes self.
                    MaybeDeviceZoneScope("wr_mcast_send");
                    sender.send(rstd_base, rstd_base, rows_t * stat_tile_bytes);
                }
                cb_push_back(cb_rstd, rows_t);
                cb_pop_front(cb_stat_partial, rows_t);
                store_block(b, rows_t);
            }
        } else {
            auto receiver = mc.receiver(noc);
            for (uint32_t b = 0; b < num_blocks; ++b) {
                const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
                gather_partials(b, rows_t);
                cb_reserve_back(cb_rstd, rows_t);
                const uint32_t rstd_base = get_write_ptr(cb_rstd);
                (void)contribute_rstd(rows_t, rstd_base);
                {
                    MaybeDeviceZoneScope("wr_mcast_recv");
                    receiver.receive();
                }
                cb_push_back(cb_rstd, rows_t);
                cb_pop_front(cb_stat_partial, rows_t);
                store_block(b, rows_t);
            }
        }
        return;
    }

    // The two mcast faces are constructed ONCE, outside the block loop: the
    // ReceiverPipe ctor kernel-inits its data_ready cell, which must not be
    // re-run after the sender has started broadcasting.
    if (is_root) {
        auto sender = mc.sender(noc);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            gather_partials(b, rows_t);
            gather_rows(b, rows_t);
            cb_reserve_back(cb_rstd, rows_t);
            const uint32_t rstd_dst = get_write_ptr(cb_rstd);
            {
                // Starved on the root's own combine + finalize chain.
                MaybeDeviceZoneScope("wr_rstd_send_wait");
                cb_wait_front(cb_rstd_send, rows_t);
            }
            {
                // src != dst selects INCLUDE_SRC loopback, so the root lands its own
                // copy in cb_rstd through the same path as every other member.
                MaybeDeviceZoneScope("wr_mcast_send");
#if RMSN_ABLATE_CROSSCORE
                (void)rstd_dst;
#else
                // Column-valid trim (Perf 1, I2): the LAST tile's face 3 is trailing
                // garbage nobody reads, and SenderPipe's one contiguous byte count is
                // exactly able to drop it. The interior faces 1/3 must still ride along
                // (a single transfer cannot skip them), which is why the multicast keeps
                // ~3/4 of the payload while the gather legs keep 1/2.
                sender.send(get_read_ptr(cb_rstd_send), rstd_dst, rows_t * stat_tile_bytes - (stat_tile_bytes >> 2));
#endif
            }
            cb_pop_front(cb_rstd_send, rows_t);
            cb_push_back(cb_rstd, rows_t);
            store_block(b, rows_t);
        }
    } else {
        auto receiver = mc.receiver(noc);
        for (uint32_t b = 0; b < num_blocks; ++b) {
            const uint32_t rows_t = (b + 1 == num_blocks) ? last_block_row_tiles : block_row_tiles;
            gather_partials(b, rows_t);
            gather_rows(b, rows_t);
            cb_reserve_back(cb_rstd, rows_t);
            {
                // The member's view of the WHOLE combine round: it blocks here
                // until the root has gathered, summed, finalized and broadcast.
                MaybeDeviceZoneScope("wr_mcast_recv");
#if !RMSN_ABLATE_CROSSCORE
                receiver.receive();
#endif
            }
            // NOTE: cb_stat_partial is popped by gather_partials on this path — it
            // PUSHES its partial to the leader, so the tiles are free as soon as the
            // write barrier retires. Do not pop again here; the split path's pull
            // gather is the only one that has to defer the release.
            cb_push_back(cb_rstd, rows_t);
            store_block(b, rows_t);
        }
    }
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// rms_norm reader (NoC0).  Realizes op_design.md's `prepare_stat_constants`,
// `load_gamma_once`, `load_block`, and the *reader half* of `combine_block`
// (the gather landing + the stat multicast).
//
// Raw-API notes (deviations from "prefer helpers"):
//  * `load_block` / `load_gamma_once` use TensorAccessor + noc_async_read
//    directly.  There is no kernel_lib helper for interleaved DRAM page
//    addressing — TensorAccessor *is* the sanctioned mechanism.  The ROW_MAJOR
//    stick path does NOT use `read_sticks_for_tilize` because that helper owns
//    its own reserve/push cycle and cannot zero the W-tail of a partial slice
//    (folding stale L1 into Sum(x^2) is a silent numeric bug); the tail is
//    zeroed here with the NoC zero engine instead of a RISC store loop.
//  * The gather landing is a raw semaphore wait: `mcast_pipe` is a *broadcast*
//    (one source -> a rectangle, one common dst address); the gather is the
//    opposite shape (s different sources -> s different destination pages on
//    one core) and kernel_lib has no gather/scatter helper.
//  * The stat broadcast DOES use the helper (`McastArgs` -> SenderPipe /
//    ReceiverPipe) rather than raw noc_async_write_multicast + semaphores.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/tensor/noc_traits.h"
#include "hostdevcommon/common_values.hpp"

#include "ttnn/cpp/ttnn/kernel_lib/mcast_pipe.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

using namespace dataflow_kernel_lib;

// Semantic CB names (the numeric slot is only the buffer index).
constexpr uint32_t cb_input_tiles = 0;
constexpr uint32_t cb_gamma_tiles = 1;
constexpr uint32_t cb_gathered_partials = 4;
constexpr uint32_t cb_rms_bcast = 5;
constexpr uint32_t cb_rms_recip = 6;
constexpr uint32_t cb_scaler = 7;
constexpr uint32_t cb_w_mask = 8;
constexpr uint32_t cb_rm_stage_in = 10;
constexpr uint32_t cb_shard_in = 13;  // ROW_MAJOR + sharded: the resident input shard

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
// A DRAM read must start on this boundary; a misaligned one silently returns the
// wrong bytes.  Only the ROW_MAJOR-sharded gamma slice can be misaligned (its
// width granule is the L1 alignment, not the tile), and it takes the hand-placed
// path below.
constexpr uint32_t DRAM_ALIGN_BYTES = 64;

// Place `n_elems` gamma values into the row-0 lanes of `tiles` consecutive tiles.
// Row 0 straddles two faces: face0 row0 at element 0, face1 row0 at 16*16.
// Everything else in the tile is already zero (NoC-zeroed by the caller), and
// BroadcastDim::Row reads row 0 only, so these are the only lanes that exist.
template <typename T>
inline void scatter_gamma_row0(
    uint32_t src_l1, uint32_t dst_l1, uint32_t tile_bytes, uint32_t n_elems, uint32_t tiles) {
    auto* src = reinterpret_cast<volatile tt_l1_ptr T*>(src_l1);
    for (uint32_t t = 0; t < tiles; ++t) {
        auto* face0 = reinterpret_cast<volatile tt_l1_ptr T*>(dst_l1 + t * tile_bytes);
        auto* face1 = face0 + FACE_DIM * FACE_DIM;
        for (uint32_t i = 0; i < FACE_DIM; ++i) {
            const uint32_t e0 = t * TILE_DIM + i;
            const uint32_t e1 = e0 + FACE_DIM;
            face0[i] = (e0 < n_elems) ? src[e0] : T(0);
            face1[i] = (e1 < n_elems) ? src[e1] : T(0);
        }
    }
}

void kernel_main() {
    // ---- mcast wire (CT 0..4, RT 0..3) ----
    constexpr auto mc = McastArgs</*CT=*/0, /*RT=*/0>();
    constexpr uint32_t CT = mc.next_compile_time_args_offset();

    // ---- block knobs (every loop bound / page count derives from these) ----
    constexpr uint32_t SLICE_HIDDEN_TILES = get_compile_time_arg_val(CT + 0);  // S
    constexpr uint32_t BLOCK_ROWS = get_compile_time_arg_val(CT + 1);          // B
    constexpr uint32_t NUM_HIDDEN_SLICES = get_compile_time_arg_val(CT + 2);   // s
    constexpr uint32_t HAS_GAMMA = get_compile_time_arg_val(CT + 3);
    constexpr uint32_t IS_ROW_MAJOR = get_compile_time_arg_val(CT + 4);
    constexpr uint32_t GAMMA_IS_TILE = get_compile_time_arg_val(CT + 5);
    constexpr uint32_t TENSOR_HIDDEN_TILES = get_compile_time_arg_val(CT + 6);  // Wt (page stride only)
    constexpr uint32_t IN_TILE_BYTES = get_compile_time_arg_val(CT + 7);
    constexpr uint32_t GAMMA_TILE_BYTES = get_compile_time_arg_val(CT + 8);
    constexpr uint32_t IN_ELEM_BYTES = get_compile_time_arg_val(CT + 9);
    constexpr uint32_t GAMMA_ELEM_BYTES = get_compile_time_arg_val(CT + 10);
    constexpr uint32_t GATHER_SEM_ID = get_compile_time_arg_val(CT + 11);
    constexpr uint32_t STAT_TILE_BYTES = get_compile_time_arg_val(CT + 12);
    constexpr uint32_t DM_CHUNK_TILES = get_compile_time_arg_val(CT + 13);
    constexpr uint32_t RM_STAGE_IN_PAGES = get_compile_time_arg_val(CT + 14);  // rm_in_depth * S
    // Gates the W-mask CB: `prepare_reduce_mask` static_asserts on cb_w_mask's
    // page FORMAT, so the call must sit in a discarded statement — not merely an
    // untaken runtime branch — on the programs where the host does not declare
    // that CB (TILE layout with W % 32 == 0, and every ROW_MAJOR program).
    constexpr uint32_t MASK_ENABLED = get_compile_time_arg_val(CT + 15);
    // Placement.  A physical shard is already resident in THIS core's L1, so the
    // input CB is bound to the caller's buffer and `load_block` degenerates to
    // publishing pages that are already there (TILE) or to core-local L1 stick
    // reads that the tilize staging needs anyway (ROW_MAJOR).  Neither path ever
    // re-reads a local shard through the TensorAccessor.
    constexpr uint32_t IS_SHARDED = get_compile_time_arg_val(CT + 16);
    // Pages compute holds at once in cb_input_tiles: one block on every path
    // except TILE+sharded, where the CB *is* the whole resident shard.
    constexpr uint32_t IN_WAIT_TILES = get_compile_time_arg_val(CT + 17);
    constexpr uint32_t SHARD_PAGE_BYTES = get_compile_time_arg_val(CT + 18);
    // TILE-layout gamma: read ONLY the two row-0 face segments of each gamma
    // tile instead of the whole 2 KB page (see the GAMMA_IS_TILE branch below).
    // Host-gated off for block-float gamma, whose tile carries a shared-exponent
    // header that a partial read would leave behind.
    constexpr uint32_t GAMMA_ROW0_ONLY = get_compile_time_arg_val(CT + 19);
    // cb_input_tiles' whole CAPACITY in pages, which is IN_WAIT_TILES only when
    // the CB holds exactly the live window.  A double-buffered input (IN_CB_DEPTH
    // > 1) makes it a multiple of the block, and the boot-time zero-fill of the
    // ragged hidden tail must cover ALL of it — every buffer is written by some
    // later block, and those tail columns are never touched again.
    constexpr uint32_t IN_CAPACITY_TILES = get_compile_time_arg_val(CT + 20);
    constexpr auto in_args = TensorAccessorArgs<CT + 21>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t RM_STICK_PITCH = SLICE_HIDDEN_TILES * TILE_DIM * IN_ELEM_BYTES;
    // Both stick offsets used below (k * RM_STICK_PITCH, and the face-1 gamma
    // relocation) must satisfy the 16 B L1 alignment.  S*32*elem is a multiple
    // of 64 for every supported elem size, so this is a guard, not a rounding.
    static_assert(RM_STICK_PITCH % 16 == 0, "ROW_MAJOR staging stick pitch must be L1-aligned");

    // DM_CHUNK_TILES is a *byte-budget* knob ("this many tiles per NoC barrier"),
    // and the ROW_MAJOR path moves sticks, not tiles.  One stick carries
    // S*32 elements = S/32 of a tile, so the same budget is DM_CHUNK_TILES*32/S
    // sticks.  Comparing the stick counter against DM_CHUNK_TILES directly (the
    // pre-fix form) barriered every 8 sticks — at S=4/bf16 that is 2 KB per
    // barrier against the TILE path's 16 KB, i.e. the same knob turned 8x finer
    // on one layout.  Clamped to a tile-row (32 sticks), which is the CB window.
    constexpr uint32_t RM_CHUNK_STICKS_RAW = (DM_CHUNK_TILES * TILE_DIM) / SLICE_HIDDEN_TILES;
    constexpr uint32_t RM_CHUNK_STICKS =
        RM_CHUNK_STICKS_RAW < 1 ? 1 : (RM_CHUNK_STICKS_RAW > TILE_DIM ? TILE_DIM : RM_CHUNK_STICKS_RAW);

    // ---- per-core runtime args ----
    constexpr uint32_t RT = mc.next_runtime_args_offset();
    const uint32_t input_addr = get_arg_val<uint32_t>(RT + 0);
    const uint32_t gamma_addr = get_arg_val<uint32_t>(RT + 1);
    const uint32_t row_start = get_arg_val<uint32_t>(RT + 2);
    const uint32_t core_row_tiles = get_arg_val<uint32_t>(RT + 3);
    const uint32_t num_blocks = get_arg_val<uint32_t>(RT + 4);
    const uint32_t slice_base = get_arg_val<uint32_t>(RT + 5);
    const uint32_t valid_tiles = get_arg_val<uint32_t>(RT + 6);
    const uint32_t valid_w = get_arg_val<uint32_t>(RT + 7);
    const uint32_t is_root = get_arg_val<uint32_t>(RT + 8);
    const uint32_t mask_valid_elems = get_arg_val<uint32_t>(RT + 9);
    const uint32_t total_sticks = get_arg_val<uint32_t>(RT + 10);
    // This core's hidden slice, in ELEMENTS.  Equals slice_base*32 everywhere
    // except a ROW_MAJOR shard, whose width granule is the L1 alignment, not 32.
    const uint32_t slice_elem_base = get_arg_val<uint32_t>(RT + 11);

    Noc noc;
    CircularBuffer cb_input_obj(cb_input_tiles);
    CircularBuffer cb_gamma_obj(cb_gamma_tiles);
    CircularBuffer cb_rm_stage_in_obj(cb_rm_stage_in);

    const auto input_accessor = TensorAccessor(in_args, input_addr);
    // Core-local NoC base (this core's own L1), used for the resident-shard
    // stick reads and for the ROW_MAJOR gamma face relocation.
    const uint64_t self_noc = get_noc_addr(my_x[noc_index], my_y[noc_index], 0);

    // ROW_MAJOR + sharded: base of the resident input shard (a CB bound to the
    // caller's L1 buffer; nothing ever pushes it, so its read pointer is the
    // shard base for the kernel's life).
    uint32_t shard_in_base = 0;
    if constexpr (IS_SHARDED && IS_ROW_MAJOR) {
        shard_in_base = get_read_ptr(cb_shard_in);
    }

    // =====================================================================
    // prepare_stat_constants — once per kernel
    // =====================================================================
    // PoolType::SUM => scaler value 1.0; 1/W and epsilon are applied once, in
    // the compute kernel's post-reduce finalize (never via PoolType::AVG, whose
    // scaler would divide by the PADDED tile width).
    calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
    if constexpr (MASK_ENABLED) {
        if (mask_valid_elems != 0) {
            prepare_reduce_mask<cb_w_mask, ReduceDim::REDUCE_ROW>(mask_valid_elems);
        }
    }

    // =====================================================================
    // load_gamma_once — this core's hidden slice, resident for the kernel
    // =====================================================================
    if constexpr (HAS_GAMMA) {
        const auto gamma_accessor = TensorAccessor(gamma_args, gamma_addr);
        cb_reserve_back(cb_gamma_tiles, SLICE_HIDDEN_TILES);
        const uint32_t gamma_l1 = get_write_ptr(cb_gamma_tiles);

        // Zero the whole slice first: a ragged last tile and any slice tiles past
        // the tensor's hidden extent must contribute gamma == 0, never stale L1.
        noc.async_write_zeros(cb_gamma_obj, SLICE_HIDDEN_TILES * GAMMA_TILE_BYTES);
        noc.write_zeros_l1_barrier();

        if constexpr (GAMMA_IS_TILE && GAMMA_ROW0_ONLY) {
            // gamma is a [W] vector, so a TILE-layout gamma is a (1, ..., 1, W)
            // tensor padded up to a whole tile-row: each of its Wt tiles carries
            // real data in ROW 0 ONLY, and the 31 rows below it are tile padding.
            // `BroadcastDim::Row` reads row 0 and nothing else (the same fact the
            // ROW_MAJOR branch below already relies on), so pulling the whole
            // 2 KB page moves 32x the bytes the consumer can see.  In the decode
            // regime (Rt == 1) that made gamma a full THIRD of the op's DRAM
            // traffic, for 1/32 of the useful payload.
            //
            // Row 0 straddles two faces: face0 row0 at tile byte 0 and face1 row0
            // at tile byte 16*16*elem.  Both offsets are multiples of 64, so both
            // reads start on a legal DRAM boundary; both landing addresses are
            // 16 B-aligned in L1.  The rest of the tile is already NoC-zeroed
            // above, which is the valid encoding for the padding rows.
            constexpr uint32_t FACE_ROW_BYTES = FACE_DIM * GAMMA_ELEM_BYTES;
            constexpr uint32_t FACE1_OFFSET = FACE_DIM * FACE_DIM * GAMMA_ELEM_BYTES;
            static_assert(FACE1_OFFSET % DRAM_ALIGN_BYTES == 0, "gamma face-1 row 0 must start on a DRAM boundary");
            for (uint32_t t = 0; t < valid_tiles; ++t) {
                const uint32_t dst = gamma_l1 + t * GAMMA_TILE_BYTES;
                noc_async_read(gamma_accessor.get_noc_addr(slice_base + t, 0), dst, FACE_ROW_BYTES);
                noc_async_read(
                    gamma_accessor.get_noc_addr(slice_base + t, FACE1_OFFSET), dst + FACE1_OFFSET, FACE_ROW_BYTES);
            }
        } else if constexpr (GAMMA_IS_TILE) {
            // Block-float gamma: the tile is not addressable row-wise (its faces
            // share an exponent header), so the whole page comes across.
            for (uint32_t t = 0; t < valid_tiles; ++t) {
                noc_async_read(
                    gamma_accessor.get_noc_addr(slice_base + t), gamma_l1 + t * GAMMA_TILE_BYTES, GAMMA_TILE_BYTES);
            }
        } else if (slice_elem_base * GAMMA_ELEM_BYTES % DRAM_ALIGN_BYTES != 0) {
            // ROW_MAJOR gamma, slice NOT on a DRAM-alignment boundary.  Only a
            // ROW_MAJOR *shard* can land here: its width granule is the L1
            // alignment (8 elements for bf16), so a width/block shard's slice can
            // start mid-DRAM-burst — and a misaligned DRAM read returns garbage
            // rather than an error.  One aligned burst covers the whole slice
            // (<= a few hundred elements, read ONCE per kernel); the 32 lanes per
            // tile that BroadcastDim::Row actually reads are then placed by hand.
            // That is the "zero over the NoC, hand-write only the real lanes"
            // pattern, not a whole-tile CPU fill: only row 0 of each face is
            // touched and the rest of the CB is already zero.
            const uint32_t byte0 = slice_elem_base * GAMMA_ELEM_BYTES;
            const uint32_t aligned_base = byte0 & ~(DRAM_ALIGN_BYTES - 1);
            const uint32_t delta = byte0 - aligned_base;
            const uint32_t scratch_l1 = gamma_l1 + SLICE_HIDDEN_TILES * GAMMA_TILE_BYTES;
            noc_async_read(
                gamma_accessor.get_noc_addr(0, aligned_base), scratch_l1, delta + valid_w * GAMMA_ELEM_BYTES);
            noc_async_read_barrier();
            if constexpr (GAMMA_ELEM_BYTES == 4) {
                scatter_gamma_row0<uint32_t>(scratch_l1 + delta, gamma_l1, GAMMA_TILE_BYTES, valid_w, valid_tiles);
            } else {
                scatter_gamma_row0<uint16_t>(scratch_l1 + delta, gamma_l1, GAMMA_TILE_BYTES, valid_w, valid_tiles);
            }
        } else {
            // ROW_MAJOR gamma is ONE stick of W elements. Only row 0 of each tile
            // is ever read (BroadcastDim::Row), and row 0 straddles two faces:
            // face0 row0 at byte 0 and face1 row0 at byte 16*16*elem.
            //
            // Step 1 lands the whole 32-element chunk at the tile's byte 0. That
            // is the ONLY DRAM offset available: a DRAM read must start on a
            // 64-byte boundary (get_dram_alignment() == 64 on Blackhole) and
            // element offset t*32 is the only multiple of 64/128 in the stick, so
            // a direct "second face" read at +16 elements is illegal. The trailing
            // 16 elements therefore spill into face0 ROW 1, which the Row
            // broadcast never reads.
            for (uint32_t t = 0; t < valid_tiles; ++t) {
                const uint32_t elems_left = valid_w - t * TILE_DIM;
                const uint32_t n = elems_left < TILE_DIM ? elems_left : TILE_DIM;
                const uint32_t src_elem = slice_elem_base + t * TILE_DIM;
                noc_async_read(
                    gamma_accessor.get_noc_addr(0, src_elem * GAMMA_ELEM_BYTES),
                    gamma_l1 + t * GAMMA_TILE_BYTES,
                    n * GAMMA_ELEM_BYTES);
            }
            noc_async_read_barrier();

            // Step 2: move the spilled 16 elements into face1 row0 with a local
            // L1->L1 NoC copy (L1 alignment is 16, so both offsets are legal).
            const uint64_t self = self_noc;
            for (uint32_t t = 0; t < valid_tiles; ++t) {
                const uint32_t elems_left = valid_w - t * TILE_DIM;
                if (elems_left <= FACE_DIM) {
                    continue;
                }
                const uint32_t rest = elems_left - FACE_DIM;
                const uint32_t n1 = rest < FACE_DIM ? rest : FACE_DIM;
                const uint32_t tile_l1 = gamma_l1 + t * GAMMA_TILE_BYTES;
                noc_async_read(
                    self + (tile_l1 + FACE_DIM * GAMMA_ELEM_BYTES),
                    tile_l1 + FACE_DIM * FACE_DIM * GAMMA_ELEM_BYTES,
                    n1 * GAMMA_ELEM_BYTES);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma_tiles, SLICE_HIDDEN_TILES);
    }

    // =====================================================================
    // Boot-time zeroing of regions the per-block reads NEVER touch.
    //
    // cb_input_tiles capacity == BLOCK_TILES exactly, so every block reuses the
    // SAME physical pages and the tail columns of a ragged hidden slice are
    // written once, at boot, and stay zero for the kernel's life.  Same for the
    // W-tail gap inside each ROW_MAJOR staging stick.
    // =====================================================================
    if constexpr (IS_ROW_MAJOR) {
        // One DM transfer, once: covers both the W-tail gap inside every staging
        // stick and any stick slot a ragged tile-row never writes.
        noc.async_write_zeros(cb_rm_stage_in_obj, RM_STAGE_IN_PAGES * IN_TILE_BYTES);
        noc.write_zeros_l1_barrier();
    } else if constexpr (IS_SHARDED) {
        // TILE + sharded: cb_input_tiles IS the caller's resident shard, so
        // `load_block` moves nothing.  What DOES have to happen once is neutralizing
        // the shard's own padding: tiles past the tensor's hidden extent, and rows
        // past its row extent, are allocated-but-never-written L1 that Sum(x^2)
        // would otherwise fold into a real row's denominator.
        constexpr uint32_t SHARD_ROWS = IN_WAIT_TILES / SLICE_HIDDEN_TILES;
        bool zeroed_any = false;
        for (uint32_t r = 0; r < SHARD_ROWS; ++r) {
            const uint32_t row_base = r * SLICE_HIDDEN_TILES;
            if (r >= core_row_tiles) {
                noc.async_write_zeros(
                    cb_input_obj, SLICE_HIDDEN_TILES * IN_TILE_BYTES, {.offset_bytes = row_base * IN_TILE_BYTES});
                zeroed_any = true;
            } else if (valid_tiles < SLICE_HIDDEN_TILES) {
                noc.async_write_zeros(
                    cb_input_obj,
                    (SLICE_HIDDEN_TILES - valid_tiles) * IN_TILE_BYTES,
                    {.offset_bytes = (row_base + valid_tiles) * IN_TILE_BYTES});
                zeroed_any = true;
            }
        }
        if (zeroed_any) {
            noc.write_zeros_l1_barrier();
        }
        // Publish the whole resident shard once.  Keeping cb_input_tiles exactly
        // FULL at every block boundary is what preserves get_write_ptr() ==
        // get_read_ptr(), which the two in-place rewrites of x depend on; the
        // block loop below re-publishes one block after each of compute's pops.
        cb_reserve_back(cb_input_tiles, IN_WAIT_TILES);
        cb_push_back(cb_input_tiles, IN_WAIT_TILES);
    } else if (valid_tiles < SLICE_HIDDEN_TILES) {
        // Every tile-row slot of the WHOLE capacity, not just the first block:
        // with IN_CB_DEPTH > 1 the reader alternates between buffers and the tail
        // columns of both must be zero before Sum(x^2) ever folds them in.
        constexpr uint32_t IN_CAPACITY_ROWS = IN_CAPACITY_TILES / SLICE_HIDDEN_TILES;
        const uint32_t pad_bytes = (SLICE_HIDDEN_TILES - valid_tiles) * IN_TILE_BYTES;
        for (uint32_t r = 0; r < IN_CAPACITY_ROWS; ++r) {
            noc.async_write_zeros(
                cb_input_obj, pad_bytes, {.offset_bytes = (r * SLICE_HIDDEN_TILES + valid_tiles) * IN_TILE_BYTES});
        }
        noc.write_zeros_l1_barrier();
    }

    // ---- combine_block wire: both faces are constructed once, outside the
    //      block loop; only the matching one is driven (is_root is per-core).
    auto sender_pipe = mc.sender(noc);
    auto receiver_pipe = mc.receiver(noc);
    Semaphore<> gather_progress(GATHER_SEM_ID);

    // =====================================================================
    // Block loop
    // =====================================================================
    for (uint32_t block = 0; block < num_blocks; ++block) {
        const uint32_t first_row = block * BLOCK_ROWS;

        if constexpr (IS_ROW_MAJOR) {
            // load_block (ROW_MAJOR): 32 sticks per tile-row into the staging CB,
            // each stick holding exactly this core's hidden slice.
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                cb_reserve_back(cb_rm_stage_in, SLICE_HIDDEN_TILES);
                const uint32_t l1 = get_write_ptr(cb_rm_stage_in);
                const uint32_t local_row = first_row + r;
                if (local_row < core_row_tiles) {
                    const uint32_t stick_base = (row_start + local_row) * TILE_DIM;
                    uint32_t pending = 0;
                    for (uint32_t k = 0; k < TILE_DIM; ++k) {
                        const uint32_t stick = stick_base + k;
                        if (stick >= total_sticks) {
                            break;
                        }
                        // Sharded: the stick already lives in THIS core's L1 and the
                        // shard page holds exactly this core's columns, so the source
                        // is a core-local address at column offset 0 — no NoC hop to
                        // DRAM, no TensorAccessor.
                        const uint64_t src = IS_SHARDED
                                                 ? (self_noc + shard_in_base + stick * SHARD_PAGE_BYTES)
                                                 : input_accessor.get_noc_addr(stick, slice_elem_base * IN_ELEM_BYTES);
                        noc_async_read(src, l1 + k * RM_STICK_PITCH, valid_w * IN_ELEM_BYTES);
                        if (++pending == RM_CHUNK_STICKS) {
                            noc_async_read_barrier();
                            pending = 0;
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_rm_stage_in, SLICE_HIDDEN_TILES);
            }
        } else if constexpr (IS_SHARDED) {
            // load_block (TILE + sharded): nothing to move.  Re-publish one
            // block's worth of pages so the resident-shard CB is FULL again at
            // this block's start (see the boot push).  At the default
            // block_rows == shard_rows this loop body never runs.
            if (block > 0) {
                cb_reserve_back(cb_input_tiles, BLOCK_TILES);
                cb_push_back(cb_input_tiles, BLOCK_TILES);
            }
        } else {
            // load_block (TILE): the whole (BLOCK_ROWS x S) block, one barrier
            // per DM_CHUNK_TILES-tile burst.
            cb_reserve_back(cb_input_tiles, BLOCK_TILES);
            const uint32_t l1 = get_write_ptr(cb_input_tiles);

            bool zeroed_any = false;
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                if (first_row + r >= core_row_tiles) {
                    noc.async_write_zeros(
                        cb_input_obj,
                        valid_tiles * IN_TILE_BYTES,
                        {.offset_bytes = r * SLICE_HIDDEN_TILES * IN_TILE_BYTES});
                    zeroed_any = true;
                }
            }
            if (zeroed_any) {
                noc.write_zeros_l1_barrier();
            }

            uint32_t pending = 0;
            for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
                const uint32_t local_row = first_row + r;
                if (local_row >= core_row_tiles) {
                    continue;
                }
                const uint32_t page = (row_start + local_row) * TENSOR_HIDDEN_TILES + slice_base;
                for (uint32_t j = 0; j < valid_tiles; ++j) {
                    noc_async_read(
                        input_accessor.get_noc_addr(page + j),
                        l1 + (r * SLICE_HIDDEN_TILES + j) * IN_TILE_BYTES,
                        IN_TILE_BYTES);
                    if (++pending == DM_CHUNK_TILES) {
                        noc_async_read_barrier();
                        pending = 0;
                    }
                }
            }
            noc_async_read_barrier();
            cb_push_back(cb_input_tiles, BLOCK_TILES);
        }

        // =================================================================
        // combine_block — reader half (only when the hidden axis is split)
        // =================================================================
        if constexpr (NUM_HIDDEN_SLICES > 1) {
            if (is_root) {
                // Gather landing: every core in the rect (this one included)
                // NoC-writes its BLOCK_ROWS partials into page (row*s + c) and
                // increments the progress counter once.
                cb_reserve_back(cb_gathered_partials, NUM_HIDDEN_SLICES * BLOCK_ROWS);
                gather_progress.wait_min((block + 1) * NUM_HIDDEN_SLICES);
                cb_push_back(cb_gathered_partials, NUM_HIDDEN_SLICES * BLOCK_ROWS);

                // Broadcast the finalized rsqrt back over the rect (loopback
                // delivers to this core's own cb_rms_recip too).
                cb_wait_front(cb_rms_bcast, BLOCK_ROWS);
                cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
                sender_pipe.send(get_read_ptr(cb_rms_bcast), get_write_ptr(cb_rms_recip), BLOCK_ROWS * STAT_TILE_BYTES);
                cb_push_back(cb_rms_recip, BLOCK_ROWS);
                cb_pop_front(cb_rms_bcast, BLOCK_ROWS);
            } else {
                cb_reserve_back(cb_rms_recip, BLOCK_ROWS);
                receiver_pipe.receive();
                cb_push_back(cb_rms_recip, BLOCK_ROWS);
            }
        }
    }
}

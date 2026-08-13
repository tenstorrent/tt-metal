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

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;

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
    constexpr auto in_args = TensorAccessorArgs<CT + 15>();
    [[maybe_unused]] constexpr auto gamma_args = TensorAccessorArgs<in_args.next_compile_time_args_offset()>();

    constexpr uint32_t BLOCK_TILES = BLOCK_ROWS * SLICE_HIDDEN_TILES;
    constexpr uint32_t RM_STICK_PITCH = SLICE_HIDDEN_TILES * TILE_DIM * IN_ELEM_BYTES;

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

    Noc noc;
    CircularBuffer cb_input_obj(cb_input_tiles);
    CircularBuffer cb_gamma_obj(cb_gamma_tiles);
    CircularBuffer cb_rm_stage_in_obj(cb_rm_stage_in);

    const auto input_accessor = TensorAccessor(in_args, input_addr);

    // =====================================================================
    // prepare_stat_constants — once per kernel
    // =====================================================================
    // PoolType::SUM => scaler value 1.0; 1/W and epsilon are applied once, in
    // the compute kernel's post-reduce finalize (never via PoolType::AVG, whose
    // scaler would divide by the PADDED tile width).
    calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>();
    if (mask_valid_elems != 0) {
        prepare_reduce_mask<cb_w_mask, ReduceDim::REDUCE_ROW>(mask_valid_elems);
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

        if constexpr (GAMMA_IS_TILE) {
            for (uint32_t t = 0; t < valid_tiles; ++t) {
                noc_async_read(
                    gamma_accessor.get_noc_addr(slice_base + t), gamma_l1 + t * GAMMA_TILE_BYTES, GAMMA_TILE_BYTES);
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
                const uint32_t src_elem = (slice_base + t) * TILE_DIM;
                noc_async_read(
                    gamma_accessor.get_noc_addr(0, src_elem * GAMMA_ELEM_BYTES),
                    gamma_l1 + t * GAMMA_TILE_BYTES,
                    n * GAMMA_ELEM_BYTES);
            }
            noc_async_read_barrier();

            // Step 2: move the spilled 16 elements into face1 row0 with a local
            // L1->L1 NoC copy (L1 alignment is 16, so both offsets are legal).
            const uint64_t self = get_noc_addr(my_x[noc_index], my_y[noc_index], 0);
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
    } else if (valid_tiles < SLICE_HIDDEN_TILES) {
        const uint32_t pad_bytes = (SLICE_HIDDEN_TILES - valid_tiles) * IN_TILE_BYTES;
        for (uint32_t r = 0; r < BLOCK_ROWS; ++r) {
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
                        noc_async_read(
                            input_accessor.get_noc_addr(stick, slice_base * TILE_DIM * IN_ELEM_BYTES),
                            l1 + k * RM_STICK_PITCH,
                            valid_w * IN_ELEM_BYTES);
                        if (++pending == DM_CHUNK_TILES) {
                            noc_async_read_barrier();
                            pending = 0;
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_rm_stage_in, SLICE_HIDDEN_TILES);
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

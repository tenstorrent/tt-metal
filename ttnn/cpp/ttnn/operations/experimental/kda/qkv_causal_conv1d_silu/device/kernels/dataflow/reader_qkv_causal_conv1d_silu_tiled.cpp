// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader of the tiled qkv_causal_conv1d_silu path (TILE input). See the kernel interface and the
// slot conventions in qkv_causal_conv1d_silu_tiled_program_factory.hpp.
//
// Per step (one column block of B tiles at one tile-row mt) the reader:
//   - reads the B input tiles X_i from DRAM into x_in (prefetched ring_steps - 1 steps ahead),
//   - forms S_1, S_2, S_3 in the shift DFB with local NoC copies of X_i and of the 3-row halo P,
//   - at the first step of a unit (a run of steps in one column block), loads the taps and the halo.
// A shifted tile S_k (k = 1..3) is 8 face-row copies (design.md section 4.2):
//   S_k.F0 rows 0..k-1 <- P rows 32-k..31 (columns 0-15)    S_k.F0 rows k..15 <- X.F0 rows 0..15-k
//   S_k.F1 rows 0..k-1 <- P rows 32-k..31 (columns 16-31)   S_k.F1 rows k..15 <- X.F1 rows 0..15-k
//   S_k.F2 rows 0..k-1 <- X.F0 rows 16-k..15                S_k.F2 rows k..15 <- X.F2 rows 0..15-k
//   S_k.F3 rows 0..k-1 <- X.F1 rows 16-k..15                S_k.F3 rows k..15 <- X.F3 rows 0..15-k
// All copies are byte copies, so S_k is bit-identical to the tilized window of the ROW_MAJOR path.
//
// NoC reads carry transaction ids, so the reader waits only for what the next action needs:
//   - trid_setup: scratch zero fills, halo and history (the first step's shift copies need them).
//   - trid_taps: the taps of a unit start. They are the slowest reads (many cores read the same
//     small tap tensors at once), so they go out last and the reader waits for them only after it
//     has pushed the step's shift and x_in tiles.
//   - one trid per x_in ring chunk: the input of step s. The reader waits for step s only.
//   - trid_copy: the shift copies of the current step. The reader pushes the step (shift and x_in)
//     as soon as these copies are complete, while later steps' input reads stay in flight.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// tt-1xx only. The scratch zero fills (noc.async_write_zeros) are NoC loopback reads from
// MEM_ZEROS_BASE on Wormhole/Blackhole, so they carry the current read transaction id and the
// trid_setup barrier waits for them. On tt-2xx (Quasar) async_write_zeros is an iDMA transaction
// that a read-trid barrier does not cover; a port must add noc.write_zeros_l1_barrier() there.
// The host factory TT_FATALs on other architectures as well.
#if !defined(ARCH_WORMHOLE) && !defined(ARCH_BLACKHOLE)
#error "reader_qkv_causal_conv1d_silu_tiled: tt-1xx (Wormhole/Blackhole) only; see the zero-fill note above"
#endif

namespace qkv_conv_tiled {

// bf16 32x32 tile: 4 faces of 16 x 16, one face row = 16 x 2 B.
constexpr uint32_t tile_bytes = 2048;
constexpr uint32_t face_bytes = 512;
constexpr uint32_t face_row_bytes = 32;
constexpr uint32_t face_rows = 16;
constexpr uint32_t tap_count = 4;
constexpr uint32_t scratch_alignment = 64;
// Halo of one tile in scratch: left half (columns 0-15) at +0, right half (columns 16-31) at +128.
constexpr uint32_t halo_half_bytes = 4 * face_row_bytes;
constexpr uint32_t halo_tile_bytes = 2 * halo_half_bytes;
// Tile rows 28-31 = face 2/3 rows 12-15; tile row 29 = face 2/3 row 13.
constexpr uint32_t f2_row12 = 2 * face_bytes + 12 * face_row_bytes;  // 1408, 64 B aligned
constexpr uint32_t f3_row12 = 3 * face_bytes + 12 * face_row_bytes;  // 1920, 64 B aligned
constexpr uint32_t f2_row13 = 2 * face_bytes + 13 * face_row_bytes;  // 1440
constexpr uint32_t f3_row13 = 3 * face_bytes + 13 * face_row_bytes;  // 1952
// A ROW-broadcast unpack of a weights tile reads faces 0 and 1 only, and the multiply uses their
// row 0 only. The reader reads faces 0 and 1 of each tap tile in one 1 KB read (row 0 = the tap,
// rows 1-15 = the tap tensor's tile padding); faces 2 and 3 of the weights entries are never read.
constexpr uint32_t tap_faces_bytes = 2 * face_bytes;
// NoC read transaction ids.
constexpr uint32_t trid_input_base = 1;  // input of the step in x_in ring chunk c: trid_input_base + c
constexpr uint32_t trid_input_count = 12;
constexpr uint32_t trid_copy = 13;   // local shift copies of the current step
constexpr uint32_t trid_setup = 14;  // scratch zero fills, halo and history reads
constexpr uint32_t trid_taps = 15;   // tap reads of a unit start

}  // namespace qkv_conv_tiled

template <
    uint32_t block_tiles,
    uint32_t Mt,
    uint32_t Ct,
    uint32_t halo_offset,
    uint32_t zeros_offset,
    uint32_t state_offset>
TT_KERNEL void reader(uint32_t step_start, uint32_t step_count) {
    using namespace qkv_conv_tiled;
    constexpr uint32_t B = block_tiles;
    constexpr uint32_t step_x_bytes = B * tile_bytes;

    const auto input = TensorAccessor(tensor::input);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
#if QKV_CONV_HAS_HISTORY
    const auto history = TensorAccessor(tensor::history);
#endif
#if QKV_CONV_RETURN_STATE
    const auto new_state = TensorAccessor(tensor::new_state);
#endif
    DataflowBuffer x_in(dfb::x_in);
    DataflowBuffer shift(dfb::shift);
    DataflowBuffer weights(dfb::weights);
    Scratchpad<uint32_t> scratch(scratch::scratch);
    Noc noc;
    const uint8_t noc_id = noc.get_noc_id();

    if (step_count == 0) {
        return;
    }
    ASSERT(x_in.get_entry_size() == tile_bytes);

    // x_in is a ring of whole steps. Its depth comes from the DFB, so the manual ring below wraps
    // exactly where the DFB wraps. Step s lives in chunk (s - step_start) % ring_steps. The input of
    // step s + ring_steps - 1 is read into the chunk of step s - 1, and only after the shift copies of
    // step s (which read rows 29-31 of step s - 1 as the halo) are complete.
    const uint32_t ring_steps = x_in.get_total_num_entries() / B;
    ASSERT(x_in.get_total_num_entries() % B == 0);
    ASSERT(ring_steps >= 3 && ring_steps <= trid_input_count);
    const uint32_t prefetch_steps = ring_steps - 1;

    const uint32_t scratch_raw = scratch.get_base_address();
    const uint32_t scratch_base = (scratch_raw + scratch_alignment - 1) & ~(scratch_alignment - 1);
    const uint32_t scratch_skew = scratch_base - scratch_raw;
    const uint32_t halo = scratch_base + halo_offset;
    const uint32_t zeros = scratch_base + zeros_offset;
#if QKV_CONV_RETURN_STATE
    const uint32_t state = scratch_base + state_offset;
#endif
#if QKV_CONV_RETURN_STATE && QKV_CONV_STATE_INPLACE
    // In-place new_state (new_state is the history buffer; see the factory header). The core that
    // reads the history tiles of a column block (the owner of the block's step mt = 0) also writes
    // that block's new_state, after the history read has landed. No other core reads those history
    // tiles, so no write can overtake a read. Rows 28-31 of the block's input tiles at tile-row Mt-1
    // are read into `stage` (laid out like the halo) together with the halo; the host places stage
    // right after the state tile.
    const uint32_t stage = state + tile_bytes;
#endif
    // Local NoC copies target this core.
    const uint64_t self_noc = get_noc_addr(my_x[noc_id], my_y[noc_id], 0, noc_id);

    // The DFB is fresh, so its write pointer is the ring base.
    const uint32_t x_ring_base = x_in.get_write_ptr();
    const uint32_t x_ring_end = x_ring_base + x_in.get_total_size_bytes();
    auto next_x_chunk = [&](uint32_t chunk) {
        chunk += step_x_bytes;
        return chunk == x_ring_end ? x_ring_base : chunk;
    };
    auto input_trid = [&](uint32_t step) { return trid_input_base + (step - step_start) % ring_steps; };
    // Reads the B input tiles of `step` into `dst`, tagged with the step's transaction id.
    auto read_step_input = [&](uint32_t step, uint32_t dst) {
        noc_async_read_set_trid(input_trid(step), noc_id);
        const uint32_t page = (step % Mt) * Ct + (step / Mt) * B;
        for (uint32_t i = 0; i < B; ++i) {
            noc_async_read<NOC_MAX_BURST_SIZE>(
                input.get_noc_addr(page + i, 0, noc_id), dst + i * tile_bytes, tile_bytes, noc_id);
        }
    };

    // Address of P row 29 of tile i: halo_left + i * halo_stride (columns 0-15), halo_right + ...
    uint32_t halo_left = 0;
    uint32_t halo_right = 0;
    uint32_t halo_stride = 0;
    // Issues the halo reads of a unit start (trid_setup). The caller waits for trid_setup before the
    // shift copies of the step.
    auto issue_unit_halo = [&](uint32_t step) {
        const uint32_t mt = step % Mt;
        const uint32_t ct0 = (step / Mt) * B;
        noc_async_read_set_trid(trid_setup, noc_id);
        if (mt > 0) {
            // Rows 28-31 of the previous tile-row; P row 29 is at +32.
            const uint32_t page = (mt - 1) * Ct + ct0;
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t h = halo + i * halo_tile_bytes;
                noc_async_read(input.get_noc_addr(page + i, f2_row12, noc_id), h, halo_half_bytes, noc_id);
                noc_async_read(
                    input.get_noc_addr(page + i, f3_row12, noc_id), h + halo_half_bytes, halo_half_bytes, noc_id);
            }
            halo_left = halo + face_row_bytes;
            halo_right = halo + halo_half_bytes + face_row_bytes;
            halo_stride = halo_tile_bytes;
        } else {
#if QKV_CONV_RETURN_STATE && QKV_CONV_STATE_INPLACE
            // Rows 28-31 of tile-row Mt-1 for this block's in-place new_state (P row 29 at +32). The
            // previous unit start's new_state writes read `stage`: they must have left L1 first.
            noc_async_writes_flushed(noc_id);
            {
                const uint32_t last_page = (Mt - 1) * Ct + ct0;
                for (uint32_t i = 0; i < B; ++i) {
                    const uint32_t st = stage + i * halo_tile_bytes;
                    noc_async_read(input.get_noc_addr(last_page + i, f2_row12, noc_id), st, halo_half_bytes, noc_id);
                    noc_async_read(
                        input.get_noc_addr(last_page + i, f3_row12, noc_id),
                        st + halo_half_bytes,
                        halo_half_bytes,
                        noc_id);
                }
            }
#endif
#if QKV_CONV_HAS_HISTORY
            // Rows 0-3 of the history tile; history row 0 (= P row 29) is at +0.
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t h = halo + i * halo_tile_bytes;
                noc_async_read(history.get_noc_addr(ct0 + i, 0, noc_id), h, halo_half_bytes, noc_id);
                noc_async_read(
                    history.get_noc_addr(ct0 + i, face_bytes, noc_id), h + halo_half_bytes, halo_half_bytes, noc_id);
            }
            halo_left = halo;
            halo_right = halo + halo_half_bytes;
            halo_stride = halo_tile_bytes;
#else
            halo_left = zeros;
            halo_right = zeros;
            halo_stride = 0;
#endif
        }
    };
    // Issues the tap reads of a unit start (trid_taps) into one weights set: faces 0 and 1 of the
    // 4B tap tiles. The caller waits for trid_taps and pushes the weights. Many cores read the same
    // small tap tensors at the same time, so these reads are the slowest ones of a unit start; they
    // go out last, and the reader forms and pushes the step's shift tiles while they are in flight.
    auto issue_unit_taps = [&](uint32_t step) {
        const uint32_t ct0 = (step / Mt) * B;
        weights.reserve_back(tap_count * B);
        noc_async_read_set_trid(trid_taps, noc_id);
        const uint32_t w = weights.get_write_ptr();
        for (uint32_t i = 0; i < B; ++i) {
            const uint32_t page = ct0 + i;
            const uint32_t w0 = w + i * tile_bytes;
            constexpr uint32_t tap_stride = B * tile_bytes;
            noc_async_read(tap0.get_noc_addr(page, 0, noc_id), w0, tap_faces_bytes, noc_id);
            noc_async_read(tap1.get_noc_addr(page, 0, noc_id), w0 + tap_stride, tap_faces_bytes, noc_id);
            noc_async_read(tap2.get_noc_addr(page, 0, noc_id), w0 + 2 * tap_stride, tap_faces_bytes, noc_id);
            noc_async_read(tap3.get_noc_addr(page, 0, noc_id), w0 + 3 * tap_stride, tap_faces_bytes, noc_id);
        }
    };

    // Prologue. Scratch is uninitialized: zero the zero-source region (the halo when history is
    // None) and the state tile. Then the first unit's halo, the first input reads, and the taps.
    // The zero fills need no write_zeros_l1_barrier(): on tt-1xx they are loopback NoC reads tagged
    // with trid_setup (set just below), and the trid_setup barrier of the first step waits for them
    // before any shift copy reads the zeros region or any new_state write reads the state tile.
    noc_async_read_set_trid(trid_setup, noc_id);
    noc.async_write_zeros(scratch, state_offset - zeros_offset, {.offset_bytes = scratch_skew + zeros_offset});
#if QKV_CONV_RETURN_STATE
    noc.async_write_zeros(scratch, tile_bytes, {.offset_bytes = scratch_skew + state_offset});
#endif
    issue_unit_halo(step_start);
    const uint32_t step_end = step_start + step_count;
    x_in.reserve_back(prefetch_steps * B);
    {
        uint32_t chunk = x_ring_base;
        for (uint32_t step = step_start; step < step_end && step < step_start + prefetch_steps; ++step) {
            read_step_input(step, chunk);
            chunk = next_x_chunk(chunk);
        }
    }
    issue_unit_taps(step_start);

    uint32_t x_cur = x_ring_base;
    uint32_t x_prev = x_ring_base;
    for (uint32_t step = step_start; step < step_end; ++step) {
        const uint32_t mt = step % Mt;
        const uint32_t ct0 = (step / Mt) * B;
        const bool unit_start = step == step_start || mt == 0;

        if (unit_start) {
            if (step != step_start) {
                issue_unit_halo(step);
                issue_unit_taps(step);
            }
            noc_async_read_barrier_with_trid(trid_setup, noc_id);
        } else {
            // P rows 29-31 are rows 29-31 of the previous step's input tiles, still in x_in.
            halo_left = x_prev + f2_row13;
            halo_right = x_prev + f3_row13;
            halo_stride = tile_bytes;
        }

        // The input of this step has landed.
        noc_async_read_barrier_with_trid(input_trid(step), noc_id);

        // S_1, S_2, S_3 of the B tiles. One set_state per copy size.
        shift.reserve_back(3 * B);
        noc_async_read_set_trid(trid_copy, noc_id);
        const uint32_t shift_base = shift.get_write_ptr();
        for (uint32_t k = 1; k <= 3; ++k) {
            const uint32_t head = k * face_row_bytes;
            const uint32_t tail = (face_rows - k) * face_row_bytes;
            const uint32_t halo_rows = (3 - k) * face_row_bytes;
            const uint32_t s_k = shift_base + (k - 1) * B * tile_bytes;
            noc_async_read_one_packet_set_state(self_noc, head, 0, noc_id);
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t x = x_cur + i * tile_bytes;
                const uint32_t d = s_k + i * tile_bytes;
                noc_async_read_one_packet_with_state(halo_left + i * halo_stride + halo_rows, d, 0, noc_id);
                noc_async_read_one_packet_with_state(
                    halo_right + i * halo_stride + halo_rows, d + face_bytes, 0, noc_id);
                noc_async_read_one_packet_with_state(x + tail, d + 2 * face_bytes, 0, noc_id);
                noc_async_read_one_packet_with_state(x + face_bytes + tail, d + 3 * face_bytes, 0, noc_id);
            }
            noc_async_read_one_packet_set_state(self_noc, tail, 0, noc_id);
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t x = x_cur + i * tile_bytes;
                const uint32_t d = s_k + i * tile_bytes;
                noc_async_read_one_packet_with_state(x, d + head, 0, noc_id);
                noc_async_read_one_packet_with_state(x + face_bytes, d + face_bytes + head, 0, noc_id);
                noc_async_read_one_packet_with_state(x + 2 * face_bytes, d + 2 * face_bytes + head, 0, noc_id);
                noc_async_read_one_packet_with_state(x + 3 * face_bytes, d + 3 * face_bytes + head, 0, noc_id);
            }
        }
        // The step is complete: hand it to compute now, not after the next step's input lands.
        noc_async_read_barrier_with_trid(trid_copy, noc_id);
        shift.push_back(3 * B);
        x_in.push_back(B);
        if (unit_start) {
            noc_async_read_barrier_with_trid(trid_taps, noc_id);
            weights.push_back(tap_count * B);
        }

#if QKV_CONV_RETURN_STATE && QKV_CONV_STATE_INPLACE
        if (mt == 0) {
            // In-place new_state tile ct0 + i (see `stage`): this unit start's trid_setup barrier has
            // landed the history halo (read by this core only) and the stage rows. Same bytes as the
            // mt = Mt-1 write below: rows 0-2 = input tile (Mt-1, ct0+i) rows 29-31, rows 3-31 = zeros.
            constexpr uint32_t rows3 = 3 * face_row_bytes;
            constexpr uint32_t rest = face_bytes - rows3;
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t st = stage + i * halo_tile_bytes;
                const uint64_t page = new_state.get_noc_addr(ct0 + i, 0, noc_id);
                noc_async_write(st + face_row_bytes, page, rows3, noc_id);
                noc_async_write(st + halo_half_bytes + face_row_bytes, page + face_bytes, rows3, noc_id);
                noc_async_write(state + rows3, page + rows3, rest, noc_id);
                noc_async_write(state + face_bytes + rows3, page + face_bytes + rows3, rest, noc_id);
                noc_async_write(state + 2 * face_bytes, page + 2 * face_bytes, 2 * face_bytes, noc_id);
            }
            // No flush here: the next stage refill (issue_unit_halo) flushes first, and the kernel's
            // final write barrier covers the last unit.
        }
#elif QKV_CONV_RETURN_STATE
        if (mt == Mt - 1) {
            // new_state tile ct0 + i: rows 0-2 = X_i rows 29-31, rows 3-31 = zeros from the state tile.
            // The five writes cover disjoint bytes of the page, so their order does not matter.
            constexpr uint32_t rows3 = 3 * face_row_bytes;
            constexpr uint32_t rest = face_bytes - rows3;
            for (uint32_t i = 0; i < B; ++i) {
                const uint32_t x = x_cur + i * tile_bytes;
                const uint64_t page = new_state.get_noc_addr(ct0 + i, 0, noc_id);
                noc_async_write(x + f2_row13, page, rows3, noc_id);
                noc_async_write(x + f3_row13, page + face_bytes, rows3, noc_id);
                noc_async_write(state + rows3, page + rows3, rest, noc_id);
                noc_async_write(state + face_bytes + rows3, page + face_bytes + rows3, rest, noc_id);
                noc_async_write(state + 2 * face_bytes, page + 2 * face_bytes, 2 * face_bytes, noc_id);
            }
            // The read of step + ring_steps refills x_cur; the writes must have left L1 by then.
            noc_async_writes_flushed(noc_id);
        }
#endif

        // Read the input of step + prefetch_steps into the chunk of step - 1 (the halo source of this
        // step, no longer needed). The reserve waits until compute has popped step - 1.
        if (step + prefetch_steps < step_end) {
            x_in.reserve_back(prefetch_steps * B);
            read_step_input(step + prefetch_steps, step == step_start ? x_ring_end - step_x_bytes : x_prev);
        }

        x_prev = x_cur;
        x_cur = next_x_chunk(x_cur);
    }

    noc_async_write_barrier(noc_id);
    noc_async_read_set_trid(0, noc_id);
}

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Unified topk_large_indices writer. The factory selects the role per
// CreateKernel site via a compile-time define:
//
//   (no role define) -- ROW-PARALLEL output writer: stream each core's
//   materialized index rows to DRAM (contiguous for the 512 LLK window,
//   face-pair -> row-major reorder otherwise).
//
//   TOPK_TREE -- COLUMN-PARALLEL tree writer: runs on every rectangle core;
//   runtime args select this core's roles per row:
//
//   RECEIVER (num_recv > 0): for each winning level with a real partner, in
//   level order: reserve the recv CB (backpressure: capacity is exactly one
//   sequence, so this blocks until compute consumed the previous level), zero
//   the local data semaphore, bump that partner's ready semaphore, wait for
//   the data semaphore, publish the sequence to compute. Pairwise version of
//   the old 2-semaphore gather protocol; reset-before-signal ordering is kept
//   on both sides so a next-row signal can never be clobbered.
//
//   SHIPPER (do_ship): wait for compute's packed survivor (or use the
//   prefilled all--inf scratch when this slice is empty with no partners),
//   wait for the winner's ready signal, reset it, NoC-write both regions into
//   the winner's recv CB (same L1 address everywhere: the recv CB spans the
//   whole rectangle, and its pointers wrap to base every full cycle), barrier,
//   then bump the winner's data semaphore.
//
//   ROOT (!do_ship): after the receive events, stream the materialized index
//   row to DRAM exactly like the row-parallel writer.

#include "api/core_local_mem.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"

#ifdef TOPK_TREE
#include "api/dataflow/noc_semaphore.h"
#endif

namespace {

template <uint32_t source_slices_per_row, uint32_t output_slices_per_row, uint32_t slice_bytes>
FORCE_INLINE void copy_row_to_scratch(CircularBuffer& src_cb, CircularBuffer& scratch_cb, const Noc& noc) {
    static_assert(source_slices_per_row == 64 || source_slices_per_row == 128);
    static_assert(output_slices_per_row >= 1 && output_slices_per_row <= source_slices_per_row);
    constexpr uint32_t transfer_bytes = slice_bytes;
    static_assert(transfer_bytes <= NOC_MAX_BURST_SIZE);

    const uint32_t src_base = src_cb.get_read_ptr();
    const uint32_t dst_base = scratch_cb.get_write_ptr();
    const uint32_t noc_id = noc.get_noc_id();
    const auto local_src = [noc_id](uint32_t addr) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = static_cast<uint32_t>(my_x[noc_id]), .noc_y = static_cast<uint32_t>(my_y[noc_id]), .addr = addr};
    };

    noc.set_async_read_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
        UnicastEndpoint{}, transfer_bytes, local_src(src_base));

    for (uint32_t dst_slice = 0; dst_slice < output_slices_per_row; ++dst_slice) {
        // pack_untilize emits 16-element slices in face-pair order:
        // [top-left, bottom-left, top-right, bottom-right] per tile column.
        const uint32_t tile_col = dst_slice >> 2;
        const uint32_t face_col = dst_slice & 0x1;
        const uint32_t face_row_offset = (dst_slice & 0x2) ? source_slices_per_row / 2 : 0;
        const uint32_t src_slice = (2 * tile_col) + face_col + face_row_offset;
        const uint32_t src_addr = src_base + src_slice * slice_bytes;
        const uint32_t dst_addr = dst_base + dst_slice * slice_bytes;
        noc.async_read_with_state<NocOptions::DEFAULT, NOC_MAX_BURST_SIZE>(
            UnicastEndpoint{},
            CoreLocalMem<uint32_t>(dst_addr),
            transfer_bytes,
            local_src(src_addr),
            {.offset_bytes = 0});
    }
    noc.async_read_barrier();
}

template <
    uint32_t source_slices_per_row,
    uint32_t output_slices_per_row,
    uint32_t slice_bytes,
    typename TensorAccessorT>
FORCE_INLINE void issue_reordered_row_write(
    CircularBuffer& src_cb,
    CircularBuffer& scratch_cb,
    const Noc& noc,
    const TensorAccessorT& tensor,
    uint32_t row,
    uint32_t row_bytes) {
    src_cb.wait_front(1);
    scratch_cb.reserve_back(1);
    copy_row_to_scratch<source_slices_per_row, output_slices_per_row, slice_bytes>(src_cb, scratch_cb, noc);
    src_cb.pop_front(1);

    scratch_cb.push_back(1);
    scratch_cb.wait_front(1);
    noc.async_write(scratch_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

template <typename TensorAccessorT>
FORCE_INLINE void issue_contiguous_row_write(
    CircularBuffer& src_cb, const Noc& noc, const TensorAccessorT& tensor, uint32_t row, uint32_t row_bytes) {
    src_cb.wait_front(1);
    noc.async_write(src_cb, tensor, row_bytes, {.offset_bytes = 0}, {.page_id = row, .offset_bytes = 0});
}

}  // namespace

#ifdef TOPK_TREE

// ----------------------------------------------------------------------
// COLUMN-PARALLEL tree writer body
// ----------------------------------------------------------------------
void kernel_main() {
    const uint32_t num_rows = get_arg_val<uint32_t>(0);
    const uint32_t num_recv = get_arg_val<uint32_t>(1);
    // Up to 7 (x, y) physical-coordinate pairs of the partners this core
    // receives from, in level order (P <= 128 -> <= 7 levels). Offsets must
    // match the factory's partner_coords(14) writer-args block.
    uint32_t partner_x[7];
    uint32_t partner_y[7];
    for (uint32_t m = 0; m < 7; ++m) {
        partner_x[m] = get_arg_val<uint32_t>(2 + 2 * m);
        partner_y[m] = get_arg_val<uint32_t>(3 + 2 * m);
    }
    const uint32_t do_ship = get_arg_val<uint32_t>(16);
    const uint32_t winner_x = get_arg_val<uint32_t>(17);
    const uint32_t winner_y = get_arg_val<uint32_t>(18);
    const uint32_t is_empty_ship = get_arg_val<uint32_t>(19);
    const uint32_t indices_addr = get_arg_val<uint32_t>(20);
    // Multi-rectangle: this rectangle's first output row (0 on a single-rect program).
    const uint32_t start_row = get_arg_val<uint32_t>(21);

    constexpr uint32_t cb_ship_values = get_compile_time_arg_val(0);
    constexpr uint32_t cb_ship_indices = get_compile_time_arg_val(1);
    constexpr uint32_t cb_neginf_scratch = get_compile_time_arg_val(2);
    constexpr uint32_t cb_recv = get_compile_time_arg_val(3);
    constexpr uint32_t ready_sem_id = get_compile_time_arg_val(4);
    constexpr uint32_t data_sem_id = get_compile_time_arg_val(5);
    constexpr uint32_t tiles_per_sequence = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t cb_indices_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(9);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(10);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(11);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(12);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(13);
    constexpr auto indices_args = TensorAccessorArgs<14>();

    constexpr uint32_t sequence_tiles = 2 * tiles_per_sequence;
    constexpr uint32_t sequence_bytes = tiles_per_sequence * tile_bytes;

    Noc noc;
    Semaphore<> ready_sem(ready_sem_id);
    Semaphore<> data_sem(data_sem_id);
    UnicastEndpoint remote;
    CircularBuffer ship_values_cb(cb_ship_values);
    CircularBuffer ship_indices_cb(cb_ship_indices);
    CircularBuffer recv_cb(cb_recv);

    // Address symmetry: the recv CB spans the whole rectangle at one address,
    // and its pointers wrap to base after every full receive cycle, so this
    // core's own write pointer doubles as the winner's destination base.
    const uint32_t recv_values_base = recv_cb.get_write_ptr();
    const uint32_t recv_indices_base = recv_values_base + sequence_bytes;

    if (do_ship != 0 && is_empty_ship != 0) {
        CircularBuffer scratch_cb(cb_neginf_scratch);
        scratch_cb.reserve_back(sequence_tiles);
        const uint32_t scratch_base = scratch_cb.get_write_ptr();
        constexpr uint32_t sequence_words = sequence_bytes / sizeof(uint32_t);
        volatile tt_l1_ptr uint32_t* scratch = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch_base);
        for (uint32_t i = 0; i < sequence_words; ++i) {
            scratch[i] = 0xFF800000u;  // exact BF16 -inf in the FP32 DST container
        }
        for (uint32_t i = 0; i < sequence_words; ++i) {
            scratch[sequence_words + i] = 0xFFFFFFFFu;  // sentinel index
        }
    }

    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer indices_cb(cb_indices_out);

    for (uint32_t row = 0; row < num_rows; ++row) {
        // Receive events, in level order.
        for (uint32_t m = 0; m < num_recv; ++m) {
            recv_cb.reserve_back(sequence_tiles);
            data_sem.set(0);
            ready_sem.up(noc, partner_x[m], partner_y[m], 1);
            data_sem.wait(1);
            recv_cb.push_back(sequence_tiles);
        }

        if (do_ship != 0) {
            uint32_t src_values;
            uint32_t src_indices;
            if (is_empty_ship != 0) {
                CircularBuffer scratch_cb(cb_neginf_scratch);
                src_values = scratch_cb.get_write_ptr();
                src_indices = src_values + sequence_bytes;
            } else {
                ship_values_cb.wait_front(tiles_per_sequence);
                ship_indices_cb.wait_front(tiles_per_sequence);
                src_values = ship_values_cb.get_read_ptr();
                src_indices = ship_indices_cb.get_read_ptr();
            }

            ready_sem.wait(1);
            ready_sem.set(0);

            noc.async_write(
                CoreLocalMem<uint32_t>(src_values),
                remote,
                sequence_bytes,
                {.offset_bytes = 0},
                {.noc_x = winner_x, .noc_y = winner_y, .addr = recv_values_base});
            noc.async_write(
                CoreLocalMem<uint32_t>(src_indices),
                remote,
                sequence_bytes,
                {.offset_bytes = 0},
                {.noc_x = winner_x, .noc_y = winner_y, .addr = recv_indices_base});
            // Drain both source reads (WAR against the compute producer) and
            // guarantee data-before-signal at the winner.
            noc.async_write_barrier();

            data_sem.up(noc, winner_x, winner_y, 1);
            noc.async_atomic_barrier();

            if (is_empty_ship == 0) {
                ship_values_cb.pop_front(tiles_per_sequence);
                ship_indices_cb.pop_front(tiles_per_sequence);
            }
        } else {
            // Root: stream the materialized index row to DRAM.
            if constexpr (source_slices_per_row == 32) {
                issue_contiguous_row_write(indices_cb, noc, indices, start_row + row, indices_page_bytes);
                noc.async_writes_flushed();
                indices_cb.pop_front(1);
            } else {
                CircularBuffer indices_scratch_cb(cb_indices_scratch);
                issue_reordered_row_write<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                    indices_cb, indices_scratch_cb, noc, indices, start_row + row, indices_page_bytes);
                noc.async_writes_flushed();
                indices_scratch_cb.pop_front(1);
            }
        }
    }

    noc.async_write_barrier();
}

#else  // !TOPK_TREE

// ----------------------------------------------------------------------
// ROW-PARALLEL output writer body
// ----------------------------------------------------------------------
void kernel_main() {
    const uint32_t indices_addr = get_arg_val<uint32_t>(0);
    const uint32_t start_row = get_arg_val<uint32_t>(1);
    const uint32_t num_rows = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_indices = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_scratch = get_compile_time_arg_val(1);
    constexpr uint32_t indices_page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t source_slices_per_row = get_compile_time_arg_val(3);
    constexpr uint32_t output_slices_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t indices_slice_bytes = get_compile_time_arg_val(5);
    constexpr auto indices_args = TensorAccessorArgs<6>();

    const auto indices = TensorAccessor(indices_args, indices_addr, indices_page_bytes);
    CircularBuffer indices_cb(cb_indices);
    Noc noc;

    if constexpr (source_slices_per_row == 32) {
        for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
            const uint32_t row = start_row + local_row;
            issue_contiguous_row_write(indices_cb, noc, indices, row, indices_page_bytes);
            noc.async_writes_flushed();
            indices_cb.pop_front(1);
        }
    } else {
        CircularBuffer indices_scratch_cb(cb_indices_scratch);
        for (uint32_t local_row = 0; local_row < num_rows; ++local_row) {
            const uint32_t row = start_row + local_row;
            issue_reordered_row_write<source_slices_per_row, output_slices_per_row, indices_slice_bytes>(
                indices_cb, indices_scratch_cb, noc, indices, row, indices_page_bytes);

            noc.async_writes_flushed();
            indices_scratch_cb.pop_front(1);
        }
    }

    noc.async_write_barrier();
}

#endif  // TOPK_TREE

// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/core_local_mem.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_layout.hpp"

using namespace qkv_conv;

// Copies channel tiles [ct_start, ct_start + ct_count) of this rank's outgoing rows and, on the final segment's
// owner, its end rows from the TILE-layout input to the exchange buffer's row-major pages. A tile row is one face
// row in each of the left and right faces; the 64-byte aligned face-row pair around it is read, then the row is
// packed and written.
template <
    uint32_t input_wt,
    uint32_t input_ct_offset,
    uint32_t max_staged_ct,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t has_actual_end>
void stage_exchange_rows(Noc& noc, uint32_t ct_start, uint32_t ct_count) {
    const auto input = TensorAccessor(tensor::input);
    const auto exchange_buffer = TensorAccessor(tensor::predecessor_rows);
    Scratchpad<uint32_t> scratch(scratch::staging);
    const uint32_t scratch_base = scratch.get_base_address();
    const uint32_t scalar = (scratch_base + 63) / 64 * 64;
    const uint32_t pairs = scalar + 64;
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };
    const auto read_scalar = [&](const auto& accessor) {
        noc.async_read(accessor, CoreLocalMem<uint32_t>(scalar), sizeof(uint32_t), {.page_id = 0}, {});
        noc.async_read_barrier();
        return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar)[0];
    };
    const uint32_t start = read_scalar(TensorAccessor(tensor::actual_start));
    uint32_t end = 0;
    if constexpr (has_actual_end) {
        end = read_scalar(TensorAccessor(*tensor::get_token_if_present<"actual_end">()));
    }
    const auto topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
    const auto interval = kda_chronology::derive_bounds<has_actual_end>(start, end, sp_rank, sp_size, local_rows);
    const uint32_t row_count = interval.final_owner == sp_rank ? 2 * history_rows : history_rows;
    const auto source_row = [&](uint32_t r) {
        if (r < history_rows) {
            return topology.outgoing_rows_begin() + r;
        }
        const uint32_t row = interval.carried_rows_begin() + r - history_rows;
        return row < local_rows ? row : local_rows - 1;
    };

    // pairs: [row][channel tile][left, right] face-row pairs; packed: [row] row segments.
    constexpr uint32_t row_pairs_bytes = max_staged_ct * 2 * face_row_pair_bytes;
    constexpr uint32_t segment_bytes = max_staged_ct * 2 * face_row_bytes;
    const uint32_t packed = pairs + row_count * row_pairs_bytes;
    for (uint32_t r = 0; r < row_count; ++r) {
        const uint32_t row = source_row(r);
        const uint32_t pair_in_face = (row % 32) / 16 * 2 * face_bytes + (row % 16) / 2 * face_row_pair_bytes;
        for (uint32_t ct = 0; ct < ct_count; ++ct) {
            for (uint32_t side = 0; side < 2; ++side) {
                noc.async_read(
                    input,
                    scratch,
                    face_row_pair_bytes,
                    {.page_id = (row / 32) * input_wt + input_ct_offset + ct_start + ct,
                     .offset_bytes = pair_in_face + side * face_bytes},
                    {.offset_bytes = offset_of(pairs) + r * row_pairs_bytes + (ct * 2 + side) * face_row_pair_bytes});
            }
        }
    }
    noc.async_read_barrier();
    constexpr uint32_t words = face_row_bytes / sizeof(uint32_t);
    for (uint32_t r = 0; r < row_count; ++r) {
        const auto* row_pairs = reinterpret_cast<tt_l1_ptr uint32_t*>(pairs + r * row_pairs_bytes);
        auto* segment = reinterpret_cast<tt_l1_ptr uint32_t*>(packed + r * segment_bytes);
        const uint32_t within_pair = source_row(r) % 2 * words;
        for (uint32_t face_row = 0; face_row < ct_count * 2; ++face_row) {
            for (uint32_t word = 0; word < words; ++word) {
                segment[face_row * words + word] = row_pairs[face_row * 2 * words + within_pair + word];
            }
        }
        noc.async_write(
            scratch,
            exchange_buffer,
            ct_count * 2 * face_row_bytes,
            {.offset_bytes = offset_of(packed) + r * segment_bytes},
            {.page_id = outgoing_page + r, .offset_bytes = ct_start * 2 * face_row_bytes});
    }
    noc.async_write_barrier();
}

template <
    uint32_t Qt,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t block_ct,
    uint32_t Mt,
    uint32_t group,
    uint32_t exchange,
    uint32_t input_wt,
    uint32_t input_ct_offset,
    uint32_t exchange_core_x,
    uint32_t exchange_core_y,
    uint32_t max_staged_ct,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t has_actual_end>
TT_KERNEL void writer(uint32_t wi_start, uint32_t wi_count, uint32_t staged_ct_start, uint32_t staged_ct_count) {
    const auto q = TensorAccessor(tensor::q);
    const auto k = TensorAccessor(tensor::k);
    const auto v = TensorAccessor(tensor::v);
    DataflowBuffer output(dfb::output);
    Noc noc;

    // With the fabric exchange, this core first stages its share of the rows the exchange core sends. The writer has
    // nothing to write until the compute's first output, so the convolution does not wait for it.
    if constexpr (exchange) {
        if (staged_ct_count != 0) {
            stage_exchange_rows<input_wt, input_ct_offset, max_staged_ct, sp_rank, sp_size, local_rows, has_actual_end>(
                noc, staged_ct_start, staged_ct_count);
        }
        Semaphore staged(sem::staged);
        staged.up(noc, exchange_core_x, exchange_core_y, 1);
    }

    const uint32_t tile_bytes = output.get_entry_size();
    const auto write = [&](uint32_t slot, const auto& tensor, uint32_t page) {
        noc.async_write(output, tensor, tile_bytes, {.offset_bytes = slot * tile_bytes}, {.page_id = page});
    };
    const auto process = [&](uint32_t work) {
        const uint32_t mt = work % Mt;
        for (uint32_t ct0 = (work / Mt) * block_ct; ct0 < (work / Mt + 1) * block_ct; ct0 += group) {
            output.wait_front(group);
            for (uint32_t slot = 0; slot < group; ++slot) {
                const uint32_t ct = ct0 + slot;
                if (ct < Qt) {
                    write(slot, q, mt * Qt + ct);
                } else if (ct < Qt + Kt) {
                    write(slot, k, mt * Kt + ct - Qt);
                } else {
                    write(slot, v, mt * Vt + ct - Qt - Kt);
                }
            }
            noc.async_write_barrier();
            output.pop_front(group);
        }
    };
    for_each_work_item(wi_start, wi_count, Mt, exchange, process);
}

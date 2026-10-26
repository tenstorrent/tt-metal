// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/qkv_conv_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tile_constants.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "internal/local_mem/address_access.h"
#include "experimental/kernel_args.h"

using namespace qkv_conv;

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

    // With the fabric exchange, stage this core's share of the rows the exchange core sends, channel tiles
    // [ct_start, ct_start + ct_count) of this rank's outgoing rows and, on the final segment's owner, its end rows.
    // The writer has nothing to write until compute's first output, so staging first does not delay the convolution.
    if constexpr (exchange) {
        const auto stage_exchanged_rows = [&](uint32_t ct_start, uint32_t ct_count) {
            const auto input = TensorAccessor(tensor::input);
            const auto exchange_buffer = TensorAccessor(tensor::exchange_buffer);
            Scratchpad<uint32_t> scratch(scratch::staging);
            constexpr StagingScratch layout{max_staged_ct};
            // Raw: base/offset arithmetic below feeds raw reads/writes into scratch; no handle form yet.
            const uint32_t scratch_base = local_mem::detail::AddressAccess::address(scratch.local_mem());
            const uint32_t base = align_up(scratch_base, read_alignment);
            const uint32_t offset = base - scratch_base;

            const auto interval = read_chronology<has_actual_end, sp_rank, sp_size, local_rows>(noc, base);
            uint32_t source_rows[StagingScratch::rows];
            for (uint32_t i = 0; i < history_rows; ++i) {
                source_rows[i] = interval.outgoing_rows_begin() + i;
                const uint32_t end_row = interval.carried_rows_begin() + i;
                // End rows past carried_rows() are never read; keep their pages in range.
                source_rows[history_rows + i] = end_row < local_rows ? end_row : local_rows - 1;
            }
            const uint32_t row_count = interval.final_owner == sp_rank ? StagingScratch::rows : history_rows;

            // Read the aligned face rows holding each row's left and right face row, pack the row's segment, then write
            // it to the exchange buffer.
            for (uint32_t r = 0; r < row_count; ++r) {
                const uint32_t row = source_rows[r];
                const uint32_t row_in_tile = row % tt::constants::TILE_HEIGHT;
                const uint32_t face_offset = (row_in_tile / face_rows) * 2 * face_bytes +
                                             (row_in_tile % face_rows) / face_rows_per_read * read_alignment;
                for (uint32_t ct = 0; ct < ct_count; ++ct) {
                    const uint32_t page =
                        (row / tt::constants::TILE_HEIGHT) * input_wt + input_ct_offset + ct_start + ct;
                    for (uint32_t side = 0; side < 2; ++side) {
                        noc.async_read(
                            input,
                            scratch,
                            read_alignment,
                            {.page_id = page, .offset_bytes = face_offset + side * face_bytes},
                            {.offset_bytes = offset + layout.read(r, ct, side)});
                    }
                }
            }
            noc.async_read_barrier();
            constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);
            for (uint32_t r = 0; r < row_count; ++r) {
                const uint32_t row_in_read = (source_rows[r] % face_rows_per_read) * face_row_bytes;
                auto* segment = reinterpret_cast<tt_l1_ptr uint32_t*>(base + layout.packed(r));
                for (uint32_t ct = 0; ct < ct_count; ++ct) {
                    for (uint32_t side = 0; side < 2; ++side) {
                        const auto* face_row =
                            reinterpret_cast<tt_l1_ptr uint32_t*>(base + layout.read(r, ct, side) + row_in_read);
                        for (uint32_t word = 0; word < face_row_words; ++word) {
                            segment[(ct * 2 + side) * face_row_words + word] = face_row[word];
                        }
                    }
                }
                noc.async_write(
                    scratch,
                    exchange_buffer,
                    ct_count * tile_row_bytes,
                    {.offset_bytes = offset + layout.packed(r)},
                    {.page_id = outgoing_page + r, .offset_bytes = ct_start * tile_row_bytes});
            }
            noc.async_write_barrier();
        };
        if (staged_ct_count != 0) {
            stage_exchanged_rows(staged_ct_start, staged_ct_count);
        }
        Semaphore staged(sem::staged);
        staged.up(noc, exchange_core_x, exchange_core_y, 1);
        noc.async_atomic_barrier();
    }

    const uint32_t tile_bytes = output.get_entry_size();
    const auto process = [&](uint32_t work) {
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = (work / Mt) * block_ct;
        for (uint32_t local_ct = 0; local_ct < block_ct; ++local_ct) {
            const uint32_t slot = local_ct % group;
            if (slot == 0) {
                output.wait_front(group);
            }
            const uint32_t ct = ct_start + local_ct;
            if (ct < Qt) {
                noc.async_write(output, q, tile_bytes, {.offset_bytes = slot * tile_bytes}, {.page_id = mt * Qt + ct});
            } else if (ct < Qt + Kt) {
                const uint32_t kt = ct - Qt;
                noc.async_write(output, k, tile_bytes, {.offset_bytes = slot * tile_bytes}, {.page_id = mt * Kt + kt});
            } else {
                const uint32_t vt = ct - Qt - Kt;
                noc.async_write(output, v, tile_bytes, {.offset_bytes = slot * tile_bytes}, {.page_id = mt * Vt + vt});
            }
            if (slot + 1 == group) {
                noc.async_write_barrier();
                output.pop_front(group);
            }
        }
    };
    for_each_work_item(wi_start, wi_count, Mt, exchange, process);
}

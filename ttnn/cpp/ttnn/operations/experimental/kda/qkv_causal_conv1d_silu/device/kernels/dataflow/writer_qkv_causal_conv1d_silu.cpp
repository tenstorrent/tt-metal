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
    using namespace qkv_conv;
    const auto q = TensorAccessor(tensor::q);
    const auto k = TensorAccessor(tensor::k);
    const auto v = TensorAccessor(tensor::v);
    DataflowBuffer output(dfb::output);
    Noc noc;

    // With the fabric exchange, this core first stages its share of the rows the exchange core sends: this rank's
    // outgoing rows and, on the final segment's owner, its end rows. The writer has nothing to write until the
    // compute's first output, so the convolution does not wait for it.
    if constexpr (exchange) {
        if (staged_ct_count != 0) {
            const auto input = TensorAccessor(tensor::input);
            const auto exchange_buffer = TensorAccessor(tensor::predecessor_rows);
            Scratchpad<uint32_t> scratch(scratch::staging);
            const uint32_t scratch_base = scratch.get_base_address();
            const uint32_t scalar_slot = (scratch_base + 63) / 64 * 64;
            const uint32_t pairs = scalar_slot + 64;
            const auto offset_of = [&](uint32_t address) { return address - scratch_base; };
            const auto read_scalar = [&](const auto& accessor) {
                noc.async_read(accessor, CoreLocalMem<uint32_t>(scalar_slot), sizeof(uint32_t), {.page_id = 0}, {});
                noc.async_read_barrier();
                return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scalar_slot)[0];
            };
            const uint32_t start = read_scalar(TensorAccessor(tensor::actual_start));
            uint32_t end = 0;
            if constexpr (has_actual_end) {
                end = read_scalar(TensorAccessor(*tensor::get_token_if_present<"actual_end">()));
            }
            const auto topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
            const auto interval =
                kda_chronology::derive_bounds<has_actual_end>(start, end, sp_rank, sp_size, local_rows);
            uint32_t source_rows[2 * history_rows];
            for (uint32_t i = 0; i < history_rows; ++i) {
                source_rows[i] = topology.outgoing_rows_begin() + i;
                const uint32_t end_row = interval.carried_rows_begin() + i;
                source_rows[history_rows + i] = end_row < local_rows ? end_row : local_rows - 1;
            }
            const uint32_t row_count = interval.final_owner == sp_rank ? 2 * history_rows : history_rows;
            // A tile row is one face row in each of its left and right faces; read the face-row pairs holding them,
            // pack each row's segment, then write it to the exchange buffer.
            const uint32_t packed = pairs + row_count * max_staged_ct * 2 * face_row_pair_bytes;
            const auto pair_offset = [&](uint32_t r, uint32_t ct, uint32_t side) {
                return ((r * max_staged_ct + ct) * 2 + side) * face_row_pair_bytes;
            };
            for (uint32_t r = 0; r < row_count; ++r) {
                const uint32_t row = source_rows[r];
                const uint32_t face_pair = (row % 32) / 16 * 2 * face_bytes + (row % 16) / 2 * face_row_pair_bytes;
                for (uint32_t ct = 0; ct < staged_ct_count; ++ct) {
                    const uint32_t page = (row / 32) * input_wt + input_ct_offset + staged_ct_start + ct;
                    for (uint32_t side = 0; side < 2; ++side) {
                        noc.async_read(
                            input,
                            scratch,
                            face_row_pair_bytes,
                            {.page_id = page, .offset_bytes = face_pair + side * face_bytes},
                            {.offset_bytes = offset_of(pairs) + pair_offset(r, ct, side)});
                    }
                }
            }
            noc.async_read_barrier();
            constexpr uint32_t face_row_words = face_row_bytes / sizeof(uint32_t);
            for (uint32_t r = 0; r < row_count; ++r) {
                const uint32_t within_pair = (source_rows[r] % 2) * face_row_bytes;
                auto* segment = reinterpret_cast<tt_l1_ptr uint32_t*>(packed + r * max_staged_ct * 2 * face_row_bytes);
                for (uint32_t ct = 0; ct < staged_ct_count; ++ct) {
                    for (uint32_t side = 0; side < 2; ++side) {
                        const auto* face_row =
                            reinterpret_cast<tt_l1_ptr uint32_t*>(pairs + pair_offset(r, ct, side) + within_pair);
                        for (uint32_t word = 0; word < face_row_words; ++word) {
                            segment[(ct * 2 + side) * face_row_words + word] = face_row[word];
                        }
                    }
                }
                noc.async_write(
                    scratch,
                    exchange_buffer,
                    staged_ct_count * 2 * face_row_bytes,
                    {.offset_bytes = offset_of(packed) + r * max_staged_ct * 2 * face_row_bytes},
                    {.page_id = outgoing_page + r, .offset_bytes = staged_ct_start * 2 * face_row_bytes});
            }
            noc.async_write_barrier();
        }
        Semaphore staged(sem::staged);
        staged.up(noc, exchange_core_x, exchange_core_y, 1);
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

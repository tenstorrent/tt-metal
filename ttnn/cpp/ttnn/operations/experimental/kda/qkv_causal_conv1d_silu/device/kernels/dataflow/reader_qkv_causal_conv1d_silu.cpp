// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for qkv_causal_conv1d_silu.
//
// Per channel block it builds the diagonal weight tiles compute multiplies by: tap t of channel c on the diagonal.
// Per tile it stages a 35-row strip for each face column (left: columns 0-15, right: columns 16-31): rows [0, 3) are
// the three tokens before the tile, rows [3, 35) the tile's own 32 tokens. Tap t's shifted tile is then strip rows
// [t, t + 32), four contiguous 512-byte face copies.

#include "ttnn/cpp/ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/qkv_causal_conv1d_silu_layout.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <
    uint32_t block_ct,
    uint32_t Mt,
    uint32_t group,
    uint32_t input_wt,
    uint32_t input_ct_offset,
    uint32_t exchange,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows>
TT_KERNEL void reader(uint32_t wi_start, uint32_t wi_count) {
    using namespace qkv_conv;
    constexpr uint32_t tap_count = history_rows + 1;
    constexpr uint32_t face_rows = 16;
    constexpr uint32_t tile_bytes = 4 * face_bytes;
    constexpr uint32_t block_row_bytes = block_ct * 2 * face_row_bytes;
    // A strip is rows [-1, 35): row -1 is landing space for the previous tile's fourth-last row.
    static_assert(strip_stride % 64 == 0 && strip_stride >= (1 + history_rows + 2 * face_rows) * face_row_bytes);

    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto predecessor = TensorAccessor(tensor::predecessor_rows);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer weights(dfb::weights);
    DataflowBuffer activation(dfb::act_tile);
    Scratchpad<uint16_t> scratch(scratch::act_window);
    Noc noc;
    UnicastEndpoint self;
    const auto local = [&](uint32_t address) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = address};
    };

    // Scratch, from a 64-byte aligned base: the strips [channel tile][left, right], then three carried rows of the
    // channel block, then the tap rows. Strips start one face row past the base, so rows -1, 3 and 19 are 64-byte
    // aligned for the input reads.
    const uint32_t scratch_base = scratch.get_base_address();
    const uint32_t aligned_base = (scratch_base + 63) / 64 * 64;
    const uint32_t strips = aligned_base + face_row_bytes;
    const uint32_t rows = aligned_base + block_ct * 2 * strip_stride;
    const uint32_t tap_rows = rows + history_rows * block_row_bytes;
    const auto strip = [&](uint32_t ct, uint32_t side) { return strips + (ct * 2 + side) * strip_stride; };
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };

    noc.async_read(
        TensorAccessor(tensor::actual_start), CoreLocalMem<uint32_t>(rows), sizeof(uint32_t), {.page_id = 0}, {});
    noc.async_read_barrier();
    const auto topology =
        kda_chronology::derive(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rows)[0], sp_rank, sp_size, local_rows);
    const uint32_t local_split_row = topology.local_split ? topology.head_rows : 0;
    Semaphore predecessor_ready(sem::predecessor_ready);
    bool predecessor_arrived = !exchange;

    // Diagonal weights, [tap][channel tile]: tap row 0 holds the weights, its first 16 in face 0, the rest in face 1.
    const auto load_weights = [&](uint32_t ct_start) {
        weights.reserve_back(tap_count * block_ct);
        noc.async_write_zeros(weights, tap_count * block_ct * tile_bytes);
        noc.write_zeros_l1_barrier();
        const auto read_tap = [&](const auto& tap, uint32_t t) {
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                for (uint32_t face = 0; face < 2; ++face) {
                    noc.async_read(
                        tap,
                        scratch,
                        face_row_pair_bytes,
                        {.page_id = ct_start + ct, .offset_bytes = face * face_bytes},
                        {.offset_bytes = offset_of(tap_rows) + ((t * block_ct + ct) * 2 + face) * face_row_pair_bytes});
                }
            }
        };
        read_tap(tap0, 0);
        read_tap(tap1, 1);
        read_tap(tap2, 2);
        read_tap(tap3, 3);
        noc.async_read_barrier();
        auto* diagonals = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(weights.get_write_ptr());
        const auto* values = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tap_rows);
        for (uint32_t tile = 0; tile < tap_count * block_ct; ++tile) {
            for (uint32_t i = 0; i < face_rows; ++i) {
                diagonals[tile * 1024 + i * 17] = values[tile * 64 + i];             // face 0
                diagonals[tile * 1024 + 768 + i * 17] = values[tile * 64 + 32 + i];  // face 3
            }
        }
        weights.push_back(tap_count * block_ct);
    };

    uint32_t loaded_block = ~0U;
    const auto process = [&](uint32_t work) {
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;
        if (block != loaded_block) {
            load_weights(ct_start);
            loaded_block = block;
        }

        // Strip rows [0, 3) are the three tokens before the tile. At a segment start they are the chronological
        // carry: the caller's history at the logical start, else the predecessor's rows. Otherwise they are the last
        // rows of the previous tile, read with the row before them as one aligned 128-byte read per face column.
        const bool segment_start = mt == 0 || (local_split_row != 0 && mt * 32 == local_split_row);
        if (segment_start) {
            const bool from_predecessor = topology.rank != topology.first_rank || mt != 0;
            if (from_predecessor && !predecessor_arrived) {
                predecessor_ready.wait_min(1);
                predecessor_arrived = true;
            }
            for (uint32_t row = 0; row < history_rows; ++row) {
                const auto read_row = [&](const auto& source, uint32_t page) {
                    noc.async_read(
                        source,
                        scratch,
                        block_row_bytes,
                        {.page_id = page, .offset_bytes = ct_start * 2 * face_row_bytes},
                        {.offset_bytes = offset_of(rows) + row * block_row_bytes});
                };
                from_predecessor ? read_row(predecessor, predecessor_page + row) : read_row(history, row);
            }
        }
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            const uint32_t page = mt * input_wt + input_ct_offset + ct_start + ct;
            for (uint32_t side = 0; side < 2; ++side) {
                if (!segment_start) {
                    noc.async_read(
                        input,
                        scratch,
                        4 * face_row_bytes,
                        {.page_id = page - input_wt,
                         .offset_bytes = (2 + side) * face_bytes + (face_rows - 4) * face_row_bytes},
                        {.offset_bytes = offset_of(strip(ct, side)) - face_row_bytes});
                }
                for (uint32_t half = 0; half < 2; ++half) {
                    noc.async_read(
                        input,
                        scratch,
                        face_bytes,
                        {.page_id = page, .offset_bytes = (2 * half + side) * face_bytes},
                        {.offset_bytes =
                             offset_of(strip(ct, side)) + (history_rows + half * face_rows) * face_row_bytes});
                }
            }
        }
        noc.async_read_barrier();
        if (segment_start) {
            for (uint32_t row = 0; row < history_rows; ++row) {
                for (uint32_t ct = 0; ct < block_ct; ++ct) {
                    for (uint32_t side = 0; side < 2; ++side) {
                        noc.async_read(
                            self,
                            scratch,
                            face_row_bytes,
                            local(rows + row * block_row_bytes + (ct * 2 + side) * face_row_bytes),
                            {.offset_bytes = offset_of(strip(ct, side)) + row * face_row_bytes});
                    }
                }
            }
            noc.async_read_barrier();
        }

        // Shifted tap tiles, [channel tile][tap] per group: tap t is strip rows [t, t + 32).
        for (uint32_t ct0 = 0; ct0 < block_ct; ct0 += group) {
            activation.reserve_back(group * tap_count);
            for (uint32_t i = 0; i < group; ++i) {
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
                    for (uint32_t face = 0; face < 4; ++face) {
                        noc.async_read(
                            self,
                            activation,
                            face_bytes,
                            local(strip(ct0 + i, face % 2) + (tap + (face / 2) * face_rows) * face_row_bytes),
                            {.offset_bytes = ((i * tap_count + tap) * 4 + face) * face_bytes});
                    }
                }
            }
            noc.async_read_barrier();
            activation.push_back(group * tap_count);
        }
    };
    for_each_work_item(wi_start, wi_count, Mt, exchange, process);
}

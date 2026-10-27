// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Reader for qkv_causal_conv1d_silu.
//
// The input is a TILE-layout tensor read in place (optionally a column window of a wider tensor). For every channel
// tile the reader stages a left and a right strip (qkv_causal_conv1d_silu_layout.hpp): rows [0, 3) hold the three
// tokens preceding the tile and rows [3, 35) the tile's own 32 tokens, so the tile's faces land in them with four
// contiguous face reads. Tap t reads tokens r - 3 + t, i.e. strip rows [t, t + 16) and [t + 16, t + 32): every tap's
// shifted tile is four contiguous face copies out of the two strips.

#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/qkv_conv_dataflow.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/hostdevcommon/qkv_causal_conv1d_silu_layout.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tile_constants.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/scratchpad.h"
#include "api/semaphore.h"
#include "internal/local_mem/address_access.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

using namespace qkv_conv;

static_assert(read_alignment % DRAM_ALIGNMENT == 0 && read_alignment % L1_ALIGNMENT == 0);

// The face loops below are unrolled by hand: the reader is the bound on wide channel blocks, and without the pragmas
// the single-block production case (48 channel tiles per block) takes 6% longer.

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
    const auto input = TensorAccessor(tensor::input);
    const auto history = TensorAccessor(tensor::history);
    const auto predecessor = TensorAccessor(tensor::predecessor_carry);
    const auto tap0 = TensorAccessor(tensor::tap0);
    const auto tap1 = TensorAccessor(tensor::tap1);
    const auto tap2 = TensorAccessor(tensor::tap2);
    const auto tap3 = TensorAccessor(tensor::tap3);
    DataflowBuffer taps(dfb::taps);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer activation(dfb::act_tile);
    Scratchpad<uint16_t> scratch(scratch::act_window);
    Noc noc;
    UnicastEndpoint self;
    const uint32_t self_x = my_x[noc.get_noc_id()];
    const uint32_t self_y = my_y[noc.get_noc_id()];
    const auto local = [&](uint32_t address) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{.noc_x = self_x, .noc_y = self_y, .addr = address};
    };

    constexpr uint32_t block_row_bytes = block_ct * tile_row_bytes;
    constexpr ReaderScratch layout{block_ct};
    // Raw: base + offset_of() arithmetic below feeds the self-loopback reads via local(); no handle form yet.
    const uint32_t scratch_base = local_mem::detail::AddressAccess::address(scratch.local_mem());
    const uint32_t base = align_up(scratch_base, read_alignment);
    const uint32_t staging = base + layout.staging();
    const uint32_t rows = base + layout.rows();
    const auto strip = [&](uint32_t parity, uint32_t ct, uint32_t side) {
        return base + strip_phase + ((parity * block_ct + ct) * 2 + side) * strip_stride;
    };
    const auto offset_of = [&](uint32_t address) { return address - scratch_base; };

    const auto topology = read_chronology<false, sp_rank, sp_size, local_rows>(noc, rows);
    const uint32_t local_split_row = topology.local_split ? topology.head_rows : 0;
    const bool initial_from_predecessor = topology.rank != topology.first_rank;

    // Read history_rows rows of this channel block's columns into the rows staging area.
    const auto read_rows = [&](const auto& source, uint32_t first_row, uint32_t ct_start) {
        for (uint32_t row = 0; row < history_rows; ++row) {
            noc.async_read(
                source,
                scratch,
                block_row_bytes,
                {.page_id = first_row + row, .offset_bytes = ct_start * tile_row_bytes},
                {.offset_bytes = offset_of(rows) + row * block_row_bytes});
        }
    };

    // With the fabric exchange, the exchange core releases this core once the predecessor's rows have landed.
    Semaphore predecessor_ready(sem::predecessor_ready);
    bool predecessor_arrived = !exchange;

    // Compute turns each tap into a diagonal tile, identity times the tap's row: it needs the identity once.
    eye.reserve_back(1);
    noc.async_write_zeros(eye, tile_bytes);
    noc.write_zeros_l1_barrier();
    constexpr uint16_t bf16_one = 0x3F80;
    auto* identity = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(eye.get_write_ptr());
    for (uint32_t face : {0, 3}) {
        for (uint32_t i = 0; i < face_rows; ++i) {
            identity[(face * face_rows + i) * face_rows + i] = bf16_one;
        }
    }
    eye.push_back(1);

    constexpr uint32_t no_block = ~0U;
    uint32_t loaded_block = no_block;
    uint32_t previous_work = no_block;
    uint32_t parity = 0;
    const auto process = [&](uint32_t work) {
        const uint32_t block = work / Mt;
        const uint32_t mt = work % Mt;
        const uint32_t ct_start = block * block_ct;

        if (block != loaded_block) {
            // The tap tiles of this channel block, [tap][channel tile]. Row 0 of a tap tile holds the weights, so only
            // its top faces are read.
            taps.reserve_back(tap_count * block_ct);
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                const auto read_tap = [&](const auto& tap, uint32_t index) {
                    noc.async_read(
                        tap,
                        taps,
                        2 * face_bytes,
                        {.page_id = ct_start + ct},
                        {.offset_bytes = (index * block_ct + ct) * tile_bytes});
                };
                read_tap(tap0, 0);
                read_tap(tap1, 1);
                read_tap(tap2, 2);
                read_tap(tap3, 3);
            }
            noc.async_read_barrier();
            taps.push_back(tap_count * block_ct);
            loaded_block = block;
        }

        // The three tokens preceding this tile: the chronological carry at a segment start (the logical start reads
        // the caller's history, any other segment start the predecessor's rows), else the previous tile's last rows.
        // Tile-aligned actual_start makes every segment start tile-aligned.
        const bool segment_start =
            mt == 0 || (local_split_row != 0 && mt * tt::constants::TILE_HEIGHT == local_split_row);
        const bool from_predecessor = segment_start && (initial_from_predecessor || mt != 0);
        const bool from_previous_strip = !segment_start && previous_work + 1 == work;
        if (from_predecessor) {
            if (!predecessor_arrived) {
                predecessor_ready.wait_min(1);
                predecessor_arrived = true;
            }
            // Without the exchange, predecessor is the bare three-row carry, read at the same pages.
            static_assert(predecessor_page == 0);
            read_rows(predecessor, predecessor_page, ct_start);
        } else if (segment_start) {
            read_rows(history, 0, ct_start);
        } else if (!from_previous_strip) {
            // The previous tile's bottom faces.
            for (uint32_t ct = 0; ct < block_ct; ++ct) {
                noc.async_read(
                    input,
                    scratch,
                    2 * face_bytes,
                    {.page_id = (mt - 1) * input_wt + input_ct_offset + ct_start + ct, .offset_bytes = 2 * face_bytes},
                    {.offset_bytes = offset_of(staging) + ct * 2 * face_bytes});
            }
        }
        // The tile's faces, into strip rows [3, 35).
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
            const uint32_t page = mt * input_wt + input_ct_offset + ct_start + ct;
#pragma GCC unroll 4
            for (uint32_t face = 0; face < 4; ++face) {
                noc.async_read(
                    input,
                    scratch,
                    face_bytes,
                    {.page_id = page, .offset_bytes = face * face_bytes},
                    {.offset_bytes = offset_of(strip(parity, ct, face % 2)) +
                                     (history_rows + (face / 2) * face_rows) * face_row_bytes});
            }
        }
        noc.async_read_barrier();

        // Strip rows [0, 3).
        constexpr uint32_t carried_bytes = history_rows * face_row_bytes;
        for (uint32_t ct = 0; ct < block_ct; ++ct) {
#pragma GCC unroll 2
            for (uint32_t side = 0; side < 2; ++side) {
                const uint32_t dst = offset_of(strip(parity, ct, side));
                if (segment_start) {
                    for (uint32_t row = 0; row < history_rows; ++row) {
                        noc.async_read(
                            self,
                            scratch,
                            face_row_bytes,
                            local(rows + row * block_row_bytes + (ct * 2 + side) * face_row_bytes),
                            {.offset_bytes = dst + row * face_row_bytes});
                    }
                } else {
                    // The last three rows of the previous tile's bottom face on this side.
                    const uint32_t previous = from_previous_strip
                                                  ? strip(parity ^ 1, ct, side) + strip_rows * face_row_bytes
                                                  : staging + ct * 2 * face_bytes + side * face_bytes + face_bytes;
                    noc.async_read(
                        self, scratch, carried_bytes, local(previous - carried_bytes), {.offset_bytes = dst});
                }
            }
        }
        noc.async_read_barrier();

        // Shifted tap tiles, [channel tile][tap] per group: face f of tap t is strip f % 2 from row t + 16 * (f / 2).
        for (uint32_t ct0 = 0; ct0 < block_ct; ct0 += group) {
            activation.reserve_back(group * tap_count);
            for (uint32_t i = 0; i < group; ++i) {
                for (uint32_t tap = 0; tap < tap_count; ++tap) {
#pragma GCC unroll 4
                    for (uint32_t face = 0; face < 4; ++face) {
                        noc.async_read(
                            self,
                            activation,
                            face_bytes,
                            local(strip(parity, ct0 + i, face % 2) + (tap + (face / 2) * face_rows) * face_row_bytes),
                            {.offset_bytes = (i * tap_count + tap) * tile_bytes + face * face_bytes});
                    }
                }
            }
            noc.async_read_barrier();
            activation.push_back(group * tap_count);
        }

        previous_work = work;
        parity ^= 1;
    };
    for_each_work_item(wi_start, wi_count, Mt, exchange, process);
}

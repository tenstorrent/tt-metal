// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"

#include <cstdint>

#include "api/tile_constants.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

// Fill one tile of `ones` with `ones_bits` and one tile of `tril` with `value_bits` on and below the diagonal (an
// inclusive lower-triangular matrix). A face row seeded in L1 is replicated with local NoC reads: down its face, and
// the first face across the tile.
// One out-of-line copy serves both fills; it runs once per program, and a copy per value overflows the kernel config
// buffer.
template <typename DFB0, typename DFB1>
__attribute__((noinline)) void fill_ones_and_tril(DFB0& ones, DFB1& tril, uint32_t value_bits, uint32_t ones_bits) {
    constexpr uint32_t face_height = tt::constants::FACE_HEIGHT;
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    constexpr uint32_t faces_per_tile = tt::constants::TILE_HW / face_elements;
    constexpr uint32_t faces_per_tile_row = tt::constants::TILE_WIDTH / face_width;
    constexpr uint32_t row_bytes = face_width * sizeof(uint32_t);
    constexpr uint32_t face_bytes = face_elements * sizeof(uint32_t);
    constexpr uint32_t lower_left_face = faces_per_tile_row;
    constexpr uint32_t bottom_right_face = faces_per_tile_row + 1;

    tril.reserve_back(1);
    ones.reserve_back(1);
    Noc noc;
    UnicastEndpoint self;
    const auto local = [&](uint32_t address) {
        return noc_traits_t<UnicastEndpoint>::src_args_type{
            .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = address};
    };
    // Seed the first row of a face with `bits`, then replicate it down the face.
    const auto fill_face = [&](auto& buffer, uint32_t face, uint32_t bits) {
        auto* row = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(buffer.get_write_ptr()) + face * face_elements;
        for (uint32_t column = 0; column < face_width; ++column) {
            row[column] = bits;
        }
        const auto source = local(buffer.get_write_ptr() + face * face_bytes);
        for (uint32_t r = 1; r < face_height; ++r) {
            noc.async_read(self, buffer, row_bytes, source, {.offset_bytes = face * face_bytes + r * row_bytes});
        }
    };

    noc.async_write_zeros(tril, tril.get_entry_size());
    fill_face(ones, 0, ones_bits);
    noc.async_read_barrier();
    const auto ones_source = local(ones.get_write_ptr());
    for (uint32_t face = 1; face < faces_per_tile; ++face) {
        noc.async_read(self, ones, face_bytes, ones_source, {.offset_bytes = face * face_bytes});
    }
    noc.write_zeros_l1_barrier();
    // The lower-left face is dense; the diagonal faces are lower-triangular, the first filled in place and copied
    // to the last.
    fill_face(tril, lower_left_face, value_bits);
    auto* tril_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tril.get_write_ptr());
    for (uint32_t row = 0; row < face_height; ++row) {
        for (uint32_t column = 0; column <= row; ++column) {
            tril_tile[row * face_width + column] = value_bits;
        }
    }
    noc.async_read_barrier();
    noc.async_read(
        self, tril, face_bytes, local(tril.get_write_ptr()), {.offset_bytes = bottom_right_face * face_bytes});
    noc.async_read_barrier();
    tril.push_back(1);
    ones.push_back(1);
}

// The identity and the nested inverse's block masks, which compute reads last.
template <typename DFB0, typename DFB1>
inline void fill_inverse_constants(DFB0& eye, DFB1& block_masks) {
    constexpr uint32_t fp32_one_bits = __builtin_bit_cast(uint32_t, 1.0F);
    constexpr uint32_t fp32_minus_one_bits = __builtin_bit_cast(uint32_t, -1.0F);
    constexpr uint32_t tile_height = tt::constants::TILE_HEIGHT;
    constexpr uint32_t tile_width = tt::constants::TILE_WIDTH;
    constexpr uint32_t tile_elements = tt::constants::TILE_HW;
    constexpr uint32_t face_height = tt::constants::FACE_HEIGHT;
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    constexpr uint32_t faces_per_tile_row = tile_width / face_width;
    constexpr uint32_t inverse_block_size = 4;
    constexpr uint32_t inverse_pair_size = 2 * inverse_block_size;
    constexpr uint32_t mask_tile_count = 3;
    constexpr uint32_t face_bytes = face_elements * sizeof(uint32_t);

    eye.reserve_back(1);
    block_masks.reserve_back(mask_tile_count);
    Noc noc;
    noc.async_write_zeros(eye, eye.get_entry_size());
    noc.async_write_zeros(block_masks, mask_tile_count * block_masks.get_entry_size());
    noc.write_zeros_l1_barrier();

    // The nested inverse consumes one mask per level, each -1 on the strictly lower entries it extracts from the raw
    // Akk (so the product is that level's part of N = -strictly_lower(Akk)): those of the 4-row diagonal blocks it
    // inverts directly, the block-lower part that pairs those blocks into 8-row blocks, and the block-lower part that
    // joins the 8-row blocks.
    // Store only the set entries, row by row: in row r the 4-row block's strictly lower part is columns [4(r/4), r),
    // the pair's is [8(r/8), 4(r/4)) (empty in the pair's first block), and the joining part is [0, 8(r/8)).
    volatile tt_l1_ptr uint32_t* masks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block_masks.get_write_ptr());
    auto set_columns = [&](uint32_t mask, uint32_t row, uint32_t first, uint32_t last) {
        const uint32_t row_base =
            (row / face_height) * faces_per_tile_row * face_elements + (row % face_height) * face_width;
        for (uint32_t column = first; column < last; ++column) {
            masks[mask * tile_elements + row_base + (column / face_width) * face_elements + column % face_width] =
                fp32_minus_one_bits;
        }
    };
    for (uint32_t row = 0; row < tile_height; ++row) {
        const uint32_t block = row / inverse_block_size * inverse_block_size;
        const uint32_t pair = row / inverse_pair_size * inverse_pair_size;
        set_columns(0, row, block, row);
        set_columns(1, row, pair, block);
        set_columns(2, row, 0, pair);
    }

    volatile tt_l1_ptr uint32_t* eye_tile = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(eye.get_write_ptr());
    for (uint32_t row = 0; row < face_height; ++row) {
        eye_tile[row * face_width + row] = fp32_one_bits;
    }
    UnicastEndpoint self;
    const auto eye_face = noc_traits_t<UnicastEndpoint>::src_args_type{
        .noc_x = my_x[noc.get_noc_id()], .noc_y = my_y[noc.get_noc_id()], .addr = eye.get_write_ptr()};
    noc.async_read(self, eye, face_bytes, eye_face, {.offset_bytes = 3 * face_bytes});
    noc.async_read_barrier();

    eye.push_back(1);
    block_masks.push_back(mask_tile_count);
}

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t max_items,
    uint32_t batch_items,
    uint32_t GATE_SCALE_BITS,
    uint32_t GATE_HALF_SCALE_BITS>
TT_KERNEL void writer(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks) {
    // The writer is idle until compute's first outputs, so it generates compute's constants, in order of first use,
    // on its own NoC while the reader reads the first batch: the gate matrices (the first stage), the L2 norm's ones
    // and the causal mask, then the inverse's identity and block masks.
    // The gate's prefix-sum (lower-triangular) and sum-broadcast (all-ones) matrices carry the gate scale, so the
    // cumulative sum applies it without a separate pass over the gate (exactly for any BF16-representable scale, since
    // the HiFi4 product of BF16 g and the scale is exact in FP32). The sum-broadcast matrix carries half the scale: the
    // matmul yields the anchor G_last / 2 directly (an exact power-of-two scaling of the same FP32 sums).
    {
        static_assert(Ct == 1, "fill_ones_and_tril fills one tile per chunk matrix");
        constexpr uint32_t fp32_one_bits = __builtin_bit_cast(uint32_t, 1.0F);
        DataflowBuffer eye(dfb::eye);
        DataflowBuffer tril(dfb::tril);
        DataflowBuffer ones(dfb::ones);
        DataflowBuffer gate_tril(dfb::gate_tril);
        DataflowBuffer gate_ones(dfb::gate_ones);
        DataflowBuffer block_masks(dfb::block_masks);
        fill_ones_and_tril(gate_ones, gate_tril, GATE_SCALE_BITS, GATE_HALF_SCALE_BITS);
        fill_ones_and_tril(ones, tril, fp32_one_bits, fp32_one_bits);
        fill_inverse_constants(eye, block_masks);
    }
    DataflowBuffer control(dfb::chronology_writer);
    control.wait_front(1);
    const uint32_t valid_chunks =
        kda_chronology::load(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control.get_read_ptr())).chunk_count();
    control.pop_front(1);
    constexpr uint32_t cc = Ct * Ct;
    constexpr uint32_t ck = Ct * Kt;

    // The reader writes v_beta and k_dec_t on the other NoC.
    const auto kd_accessor = TensorAccessor(tensor::kd_output);
    const auto q_decay_accessor = TensorAccessor(tensor::q_decay_output);
    const auto intra_accessor = TensorAccessor(tensor::intra_output);
    const auto final_decay_accessor = TensorAccessor(tensor::final_decay_output);
    const auto t_inv_accessor = TensorAccessor(tensor::t_inv_output);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer final_decay(dfb::final_decay);
    DataflowBuffer t_inv(dfb::t_inv);
    Noc noc;

    // Write `items` work items' tiles, `tiles` each, published together, to their pages.
    auto drain_items =
        [&](auto& buffer, const auto& accessor, uint32_t tiles, const uint32_t* work_items, uint32_t items) {
            buffer.wait_front(items * tiles);
            for (uint32_t item = 0; item < items; ++item) {
                for (uint32_t tile = 0; tile < tiles; ++tile) {
                    noc.async_write(
                        buffer,
                        accessor,
                        buffer.get_entry_size(),
                        {.offset_bytes = (item * tiles + tile) * buffer.get_entry_size()},
                        {.page_id = work_items[item] * tiles + tile});
                }
            }
            noc.async_write_barrier();
            buffer.pop_front(items * tiles);
        };
    // Compute runs the valid work items in batches of batch_items (those past the last whole batch one at a time) and
    // publishes each output for a whole batch, in this order.
    uint32_t valid_items = 0;
    for (uint32_t index = 0; index < work_item_count; ++index) {
        if ((work_item_start + index) % num_chunks < valid_chunks) {
            ++valid_items;
        }
    }
    uint32_t next = 0;
    for (uint32_t done = 0; done < valid_items;) {
        const uint32_t items = valid_items - done >= batch_items ? batch_items : 1;
        uint32_t batch[batch_items];
        for (uint32_t item = 0; item < items; ++next) {
            if ((work_item_start + next) % num_chunks < valid_chunks) {
                batch[item++] = work_item_start + next;
            }
        }
        drain_items(final_decay, final_decay_accessor, Kt, batch, items);
        drain_items(q_decay, q_decay_accessor, ck, batch, items);
        drain_items(kd, kd_accessor, ck, batch, items);
        drain_items(intra, intra_accessor, cc, batch, items);
        done += items;
    }
    // Compute inverts every item's Akk together after the per-item terms, padded to max_items, and publishes them
    // together: write them with one barrier.
    uint32_t inverse_work_items[max_items];
    uint32_t inverse_items = 0;
    for (uint32_t index = 0; index < work_item_count; ++index) {
        const uint32_t work_item = work_item_start + index;
        if (work_item % num_chunks < valid_chunks) {
            inverse_work_items[inverse_items++] = work_item;
        }
    }
    if (inverse_items != 0) {
        drain_items(t_inv, t_inv_accessor, cc, inverse_work_items, inverse_items);
    }
    if (inverse_items != 0 && inverse_items != max_items) {
        t_inv.wait_front((max_items - inverse_items) * cc);
        t_inv.pop_front((max_items - inverse_items) * cc);
    }
}

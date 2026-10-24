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

// Issue zero writes for rows [first_row, TILE_HEIGHT) of reserved tiles [first_tile, first_tile + tiles); the
// caller waits with write_zeros_l1_barrier() before publishing the buffer.
// A tile stores its four 16x16 faces in order f0 f1 f2 f3, so for 0 < first_row its
// padded rows are two contiguous ranges: the tail of the left face (f0 or f2) in the
// face row that holds first_row, and everything from the same row of the right face
// (f1 or f3) to the end of the tile; below a padded top face row, f2 and f3 are all
// padding. Inputs are BF16 or FP32, so face rows are 32 or 64 bytes and both ranges
// start at the 16-byte alignment of the zero writes.
template <typename DFB0>
__attribute__((noinline)) void zero_rows_from(
    const Noc& noc, DFB0& buffer, uint32_t first_tile, uint32_t tiles, uint32_t first_row) {
    constexpr uint32_t face_height = tt::constants::FACE_HEIGHT;
    constexpr uint32_t face_width = tt::constants::FACE_WIDTH;
    constexpr uint32_t faces_per_tile_row = tt::constants::TILE_WIDTH / face_width;
    static_assert(faces_per_tile_row == 2, "two padded ranges per tile assume 2x2 faces");
    const uint32_t entry_bytes = buffer.get_entry_size();
    const uint32_t face_bytes = entry_bytes / (faces_per_tile_row * faces_per_tile_row);
    const uint32_t row_bytes = face_bytes / face_height;
    const uint32_t row_offset = first_row % face_height * row_bytes;
    const uint32_t left = first_row / face_height * faces_per_tile_row * face_bytes + row_offset;
    const uint32_t right = left + face_bytes;
    for (uint32_t tile = first_tile; tile < first_tile + tiles; ++tile) {
        const uint32_t tile_base = tile * entry_bytes;
        noc.async_write_zeros(buffer, face_bytes - row_offset, {.offset_bytes = tile_base + left});
        noc.async_write_zeros(buffer, entry_bytes - right, {.offset_bytes = tile_base + right});
    }
}

template <
    uint32_t Ct,
    uint32_t Kt,
    uint32_t Vt,
    uint32_t beta_width_tiles,
    uint32_t beta_offset_tiles,
    uint32_t GATE_SCALE_BITS,
    uint32_t GATE_HALF_SCALE_BITS,
    uint32_t has_actual_start,
    uint32_t has_actual_end,
    uint32_t sp_rank,
    uint32_t sp_size,
    uint32_t local_rows,
    uint32_t batch_items>
TT_KERNEL void reader(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks, uint32_t num_heads) {
    // Padding is zeroed per tile, so each tile row must be one whole chunk.
    static_assert(Ct == 1, "chunk KDA currently requires chunk_size=32");

    const auto q_accessor = TensorAccessor(tensor::q);
    const auto k_accessor = TensorAccessor(tensor::k);
    const auto v_accessor = TensorAccessor(tensor::v);
    const auto g_accessor = TensorAccessor(tensor::g);
    const auto beta_accessor = TensorAccessor(tensor::beta);
    DataflowBuffer q(dfb::q);
    DataflowBuffer k(dfb::k);
    DataflowBuffer v(dfb::v);
    DataflowBuffer g(dfb::g);
    DataflowBuffer beta(dfb::beta);
    DataflowBuffer eye(dfb::eye);
    DataflowBuffer tril(dfb::tril);
    DataflowBuffer ones(dfb::ones);
    DataflowBuffer gate_tril(dfb::gate_tril);
    DataflowBuffer gate_ones(dfb::gate_ones);
    DataflowBuffer block_masks(dfb::block_masks);
    Noc noc;

    uint32_t valid_chunks;
    uint32_t partial_chunk;
    uint32_t partial_rows;
    {
        DataflowBuffer control(dfb::chronology_compute);
        control.reserve_back(1);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control.get_write_ptr());
        uint32_t start = 0;
        if constexpr (has_actual_start) {
            const auto start_tensor = TensorAccessor(*tensor::get_token_if_present<"actual_start">());
            noc.async_read(start_tensor, control, sizeof(uint32_t), {.page_id = 0}, {});
            noc.async_read_barrier();
            start = words[0];
        }
        auto topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
        if constexpr (has_actual_end) {
            const auto end_tensor = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
            noc.async_read(end_tensor, control, sizeof(uint32_t), {.page_id = 0}, {});
            noc.async_read_barrier();
            topology = kda_chronology::derive_interval(start, words[0], sp_rank, sp_size, local_rows);
        }
        valid_chunks = topology.chunk_count();
        partial_chunk = topology.partial_chunk();
        partial_rows = topology.partial_rows();
        kda_chronology::store(words, topology);
        control.push_back(1);
        DataflowBuffer writer_control(dfb::chronology_writer);
        writer_control.reserve_back(1);
        kda_chronology::store(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_control.get_write_ptr()), topology);
        writer_control.push_back(1);
    }

    // Compute runs the valid work items in batches of batch_items (those past the last whole batch one at a time), and
    // each input buffer holds one batch. Read a batch tensor by tensor, in the order compute consumes them (g, beta,
    // q, v, k): a tensor's reads for the whole batch are in flight together and are published as soon as they land,
    // so compute starts on the gate while the rest of the batch is read. Each reservation waits for compute to
    // release that buffer's previous batch.
    uint32_t batch_work_items[batch_items];
    uint32_t batch_size = 0;
    const auto read_batch = [&](const auto& accessor, auto& buffer, uint32_t width_tiles) {
        buffer.reserve_back(batch_size * Ct * width_tiles);
        const uint32_t row_stride = num_heads * width_tiles;
        for (uint32_t item = 0; item < batch_size; ++item) {
            const uint32_t head = batch_work_items[item] / num_chunks;
            const uint32_t chunk = batch_work_items[item] % num_chunks;
            for (uint32_t row = 0; row < Ct; ++row) {
                for (uint32_t col = 0; col < width_tiles; ++col) {
                    const uint32_t page = (chunk * Ct + row) * row_stride + head * width_tiles + col;
                    noc.async_read(
                        accessor,
                        buffer,
                        buffer.get_entry_size(),
                        {.page_id = page},
                        {.offset_bytes = ((item * Ct + row) * width_tiles + col) * buffer.get_entry_size()});
                }
            }
        }
    };
    // Zero G, beta, K, and V past the valid end: zero decay and zero K and V make each padded row an identity step,
    // and replacing (not scaling) the rows keeps arbitrary padding, including NaN, out of the carries. Q only reaches
    // padded outputs. Beta logits zero to sigmoid(0) = 0.5, which is harmless: beta only scales K and V. Zero beta
    // before moving the head column, so row 0 takes the zeroed rows.
    const auto zero_padding = [&](auto& buffer, uint32_t width_tiles) {
        if (partial_rows == 0) {
            return;
        }
        bool zeroed = false;
        for (uint32_t item = 0; item < batch_size; ++item) {
            if (batch_work_items[item] % num_chunks == partial_chunk) {
                zero_rows_from(noc, buffer, item * Ct * width_tiles, Ct * width_tiles, partial_rows);
                zeroed = true;
            }
        }
        if (zeroed) {
            noc.write_zeros_l1_barrier();
        }
    };
    const auto publish = [&](auto& buffer, uint32_t width_tiles, bool zero) {
        noc.async_read_barrier();
        if (zero) {
            zero_padding(buffer, width_tiles);
        }
        buffer.push_back(batch_size * Ct * width_tiles);
    };

    // In order of first use: the gate matrices (with the first g), the L2 norm's ones and the causal mask (with the
    // first q), then the inverse's, after the first batch.
    // The gate's prefix-sum (lower-triangular) and sum-broadcast (all-ones) matrices carry the gate scale, so the
    // cumulative sum applies it without a separate pass over the gate (exactly for any BF16-representable scale, since
    // the HiFi4 product of BF16 g and the scale is exact in FP32). The sum-broadcast matrix carries half the scale: the
    // matmul yields the anchor G_last / 2 directly (an exact power-of-two scaling of the same FP32 sums).
    static_assert(Ct == 1, "fill_ones_and_tril fills one tile per chunk matrix");
    constexpr uint32_t fp32_one_bits = __builtin_bit_cast(uint32_t, 1.0F);
    bool first_batch = true;
    uint32_t next = 0;
    uint32_t valid_items = 0;
    for (uint32_t index = 0; index < work_item_count; ++index) {
        if ((work_item_start + index) % num_chunks < valid_chunks) {
            ++valid_items;
        }
    }
    for (uint32_t done = 0; done < valid_items;) {
        batch_size = valid_items - done >= batch_items ? batch_items : 1;
        for (uint32_t item = 0; item < batch_size; ++next) {
            if ((work_item_start + next) % num_chunks < valid_chunks) {
                batch_work_items[item++] = work_item_start + next;
            }
        }

        read_batch(g_accessor, g, Kt);
        if (first_batch) {
            fill_ones_and_tril(gate_ones, gate_tril, GATE_SCALE_BITS, GATE_HALF_SCALE_BITS);
        }
        publish(g, Kt, true);

        // Token-major beta logits: the chunk's tile holds 32 heads' logits as columns.
        static_assert(Ct == 1, "token-major beta supports one tile row per chunk");
        beta.reserve_back(batch_size * Ct);
        for (uint32_t item = 0; item < batch_size; ++item) {
            const uint32_t head = batch_work_items[item] / num_chunks;
            const uint32_t chunk = batch_work_items[item] % num_chunks;
            noc.async_read(
                beta_accessor,
                beta,
                beta.get_entry_size(),
                {.page_id = chunk * beta_width_tiles + beta_offset_tiles + head / tt::constants::TILE_WIDTH},
                {.offset_bytes = item * beta.get_entry_size()});
        }
        noc.async_read_barrier();
        zero_padding(beta, 1);
        // Compute activates row 0 (fewer SFPU iterations than a column) and transposes it to the column 0 the
        // broadcasts read; move this head's BF16 logit column into row 0. Token r's logit lands in row 0, column r.
        // The only source in row 0 is token 0's, read first, so the in-place move is safe.
        for (uint32_t item = 0; item < batch_size; ++item) {
            auto* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(beta.get_write_ptr()) +
                         item * beta.get_entry_size() / sizeof(uint16_t);
            const uint32_t column = batch_work_items[item] / num_chunks % tt::constants::TILE_WIDTH;
            const uint32_t column_face = column / tt::constants::FACE_WIDTH;
            const uint32_t column_offset = column % tt::constants::FACE_WIDTH;
            for (uint32_t row = 0; row < tt::constants::TILE_HEIGHT; ++row) {
                const uint32_t face_row = row / tt::constants::FACE_HEIGHT;
                const uint32_t row_base = (row % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
                const uint32_t source =
                    (face_row * 2 + column_face) * tt::constants::FACE_HW + row_base + column_offset;
                // Row 0, column `row`: face 0 or 1, first face row.
                const uint32_t target =
                    row / tt::constants::FACE_WIDTH * tt::constants::FACE_HW + row % tt::constants::FACE_WIDTH;
                tile[target] = tile[source];
            }
        }
        beta.push_back(batch_size * Ct);

        read_batch(q_accessor, q, Kt);
        if (first_batch) {
            fill_ones_and_tril(ones, tril, fp32_one_bits, fp32_one_bits);
        }
        publish(q, Kt, false);
        read_batch(v_accessor, v, Vt);
        publish(v, Vt, true);
        read_batch(k_accessor, k, Kt);
        publish(k, Kt, true);

        if (first_batch) {
            fill_inverse_constants(eye, block_masks);
            first_batch = false;
        }
        done += batch_size;
    }
    if (first_batch) {
        // No valid work item: compute still awaits the gate matrices up front.
        fill_ones_and_tril(gate_ones, gate_tril, GATE_SCALE_BITS, GATE_HALF_SCALE_BITS);
        fill_ones_and_tril(ones, tril, fp32_one_bits, fp32_one_bits);
        fill_inverse_constants(eye, block_masks);
    }
}

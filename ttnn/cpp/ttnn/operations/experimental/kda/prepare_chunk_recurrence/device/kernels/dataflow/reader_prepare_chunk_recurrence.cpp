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

    // The nested inverse consumes one mask per level: the 4-row diagonal blocks it inverts
    // directly, the strict block-lower part that pairs those blocks into 8-row blocks, and the
    // strict block-lower part that joins the 8-row blocks.
    // Store only the set entries, row by row: in row r the 4-row block is columns [4(r/4), 4(r/4)+4), the rest of
    // the 8-row pair is the other 4 columns of [8(r/8), 8(r/8)+8), and the strict block-lower part is [0, 8(r/8)).
    volatile tt_l1_ptr uint32_t* masks = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(block_masks.get_write_ptr());
    auto set_columns = [&](uint32_t mask, uint32_t row, uint32_t first, uint32_t last) {
        const uint32_t row_base =
            (row / face_height) * faces_per_tile_row * face_elements + (row % face_height) * face_width;
        for (uint32_t column = first; column < last; ++column) {
            masks[mask * tile_elements + row_base + (column / face_width) * face_elements + column % face_width] =
                fp32_one_bits;
        }
    };
    for (uint32_t row = 0; row < tile_height; ++row) {
        const uint32_t block = row / inverse_block_size * inverse_block_size;
        const uint32_t pair = row / inverse_pair_size * inverse_pair_size;
        set_columns(0, row, block, block + inverse_block_size);
        set_columns(1, row, pair, pair + inverse_pair_size);
        set_columns(2, row, 0, pair);
    }
    // Mask 1 holds the pair minus the block: clear the block entries just written.
    for (uint32_t row = 0; row < tile_height; ++row) {
        const uint32_t block = row / inverse_block_size * inverse_block_size;
        const uint32_t row_base =
            (row / face_height) * faces_per_tile_row * face_elements + (row % face_height) * face_width;
        for (uint32_t column = block; column < block + inverse_block_size; ++column) {
            masks[tile_elements + row_base + (column / face_width) * face_elements + column % face_width] = 0;
        }
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

// Issue zero writes for rows [first_row, TILE_HEIGHT) of each reserved tile; the
// caller waits with write_zeros_l1_barrier() before publishing the buffer.
// A tile stores its four 16x16 faces in order f0 f1 f2 f3, so for 0 < first_row its
// padded rows are two contiguous ranges: the tail of the left face (f0 or f2) in the
// face row that holds first_row, and everything from the same row of the right face
// (f1 or f3) to the end of the tile; below a padded top face row, f2 and f3 are all
// padding. Inputs are BF16 or FP32, so face rows are 32 or 64 bytes and both ranges
// start at the 16-byte alignment of the zero writes.
template <typename DFB0>
__attribute__((noinline)) void zero_rows_from(const Noc& noc, DFB0& buffer, uint32_t tiles, uint32_t first_row) {
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
    for (uint32_t tile = 0; tile < tiles; ++tile) {
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
    uint32_t local_rows>
TT_KERNEL void reader(uint32_t work_item_start, uint32_t work_item_count, uint32_t num_chunks, uint32_t num_heads) {
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t chunk_value_tiles = Ct * Vt;
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

    auto enqueue_contiguous_read = [&](const auto& accessor, auto& buffer, uint32_t base, uint32_t tiles) {
        buffer.reserve_back(tiles);
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            noc.async_read(
                accessor,
                buffer,
                buffer.get_entry_size(),
                {.page_id = base + tile},
                {.offset_bytes = tile * buffer.get_entry_size()});
        }
    };
    // The constants are filled while the first work item's reads are in flight.
    bool constants_pending = true;
    const auto fill_constants = [&]() {
        // In order of first use: the L2 norm's ones (and the causal mask), the gate matrices, then the inverse's.
        constexpr uint32_t fp32_one_bits = __builtin_bit_cast(uint32_t, 1.0F);
        fill_ones_and_tril(ones, tril, fp32_one_bits, fp32_one_bits);
        // The gate's prefix-sum (lower-triangular) and sum-broadcast (all-ones) matrices carry the gate scale, so
        // the cumulative sum applies it without a separate pass over the gate (exactly for any BF16-representable
        // scale, since the HiFi4 product of BF16 g and the scale is exact in FP32).
        static_assert(Ct == 1, "fill_ones_and_tril fills one tile per chunk matrix");
        // The sum-broadcast matrix carries half the scale: the matmul yields the anchor G_last / 2 directly (an exact
        // power-of-two scaling of the same FP32 sums).
        fill_ones_and_tril(gate_ones, gate_tril, GATE_SCALE_BITS, GATE_HALF_SCALE_BITS);
        fill_inverse_constants(eye, block_masks);
        constants_pending = false;
    };

    auto enqueue_head_chunk_read =
        [&](const auto& accessor, auto& buffer, uint32_t head_chunk_index, uint32_t width_tiles) {
            const uint32_t head = head_chunk_index / num_chunks;
            const uint32_t chunk = head_chunk_index % num_chunks;
            const uint32_t row_stride = num_heads * width_tiles;
            buffer.reserve_back(Ct * width_tiles);
            for (uint32_t row = 0; row < Ct; ++row) {
                for (uint32_t col = 0; col < width_tiles; ++col) {
                    const uint32_t page = (chunk * Ct + row) * row_stride + head * width_tiles + col;
                    noc.async_read(
                        accessor,
                        buffer,
                        buffer.get_entry_size(),
                        {.page_id = page},
                        {.offset_bytes = (row * width_tiles + col) * buffer.get_entry_size()});
                }
            }
        };

    for (uint32_t index = 0; index < work_item_count; ++index) {
        const uint32_t head_chunk_index = work_item_start + index;
        if (head_chunk_index % num_chunks >= valid_chunks) {
            continue;
        }
        enqueue_head_chunk_read(q_accessor, q, head_chunk_index, Kt);
        enqueue_head_chunk_read(k_accessor, k, head_chunk_index, Kt);
        enqueue_head_chunk_read(v_accessor, v, head_chunk_index, Vt);
        enqueue_head_chunk_read(g_accessor, g, head_chunk_index, Kt);
        // Token-major beta logits: the chunk's tile holds 32 heads' logits as columns.
        static_assert(Ct == 1, "token-major beta supports one tile row per chunk");
        const uint32_t head = head_chunk_index / num_chunks;
        const uint32_t chunk = head_chunk_index % num_chunks;
        enqueue_contiguous_read(
            beta_accessor, beta, chunk * beta_width_tiles + beta_offset_tiles + head / tt::constants::TILE_WIDTH, 1);
        // All five inputs are independent reads on the same NoC. One barrier lets them overlap, then publishes
        // the complete work item atomically to compute.
        noc.async_read_barrier();
        // Zero G, beta, K, and V past the valid end: zero decay and zero K and V make
        // each padded row an identity step, and replacing (not scaling) the rows keeps
        // arbitrary padding, including NaN, out of the carries. Q only reaches padded outputs.
        // Beta logits zero to sigmoid(0) = 0.5, which is harmless: beta only scales K and V.
        // Zero before moving the head column, so column 0 takes the zeroed rows.
        if (partial_rows != 0 && head_chunk_index % num_chunks == partial_chunk) {
            zero_rows_from(noc, k, chunk_key_tiles, partial_rows);
            zero_rows_from(noc, v, chunk_value_tiles, partial_rows);
            zero_rows_from(noc, g, chunk_key_tiles, partial_rows);
            zero_rows_from(noc, beta, Ct, partial_rows);
            noc.write_zeros_l1_barrier();
        }
        // Compute broadcasts column 0; move this head's BF16 logit column there. Each row reads its source before
        // overwriting column 0, so the in-place move is safe.
        {
            auto* tile = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(beta.get_write_ptr());
            const uint32_t column = head % tt::constants::TILE_WIDTH;
            const uint32_t column_face = column / tt::constants::FACE_WIDTH;
            const uint32_t column_offset = column % tt::constants::FACE_WIDTH;
            for (uint32_t row = 0; row < tt::constants::TILE_HEIGHT; ++row) {
                const uint32_t face_row = row / tt::constants::FACE_HEIGHT;
                const uint32_t row_base = (row % tt::constants::FACE_HEIGHT) * tt::constants::FACE_WIDTH;
                const uint32_t source =
                    (face_row * 2 + column_face) * tt::constants::FACE_HW + row_base + column_offset;
                tile[face_row * 2 * tt::constants::FACE_HW + row_base] = tile[source];
            }
        }
        q.push_back(chunk_key_tiles);
        k.push_back(chunk_key_tiles);
        v.push_back(chunk_value_tiles);
        g.push_back(chunk_key_tiles);
        beta.push_back(Ct);
        // Compute starts on the first item while the constants are generated; it waits for each one at first use.
        if (constants_pending) {
            fill_constants();
        }
    }
    if (constants_pending) {
        fill_constants();
    }
}

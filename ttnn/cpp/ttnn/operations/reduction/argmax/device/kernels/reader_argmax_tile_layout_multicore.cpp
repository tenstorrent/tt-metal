// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Multi-core argmax over the last dim of a TILE input, reading the tile bytes.
//
// Work is a set of tile rows (32 input rows each, flattened over the outer dims) and, for each, a
// range of tile columns. A group of cores_per_group cores shares a set of tile rows and splits
// their columns. When a tile row holds more than one face of rows, two instances of this kernel run
// on every core, one on the top faces (rows 0-15) and one on the bottom faces (rows 16-31); each
// reads only its faces and has its own buffers and semaphore. Elements are compared through
// argmax_order_key.hpp, whose integer order is the single-core reader's comparator.
// With one core per group a core writes its rows' indices itself. Otherwise each core leaves one
// (key, index) pair per row in the group leader's L1, and the leader merges them in column order,
// so ties and NaNs resolve as in one left-to-right pass.

#include "argmax_tile_layout.hpp"
#include "argmax_common.hpp"
#include "argmax_order_key.hpp"
#include "api/dataflow/dataflow_api.h"
#include "api/tensor/tensor_accessor.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <stdint.h>
#include <type_traits>

namespace {

template <DataFormat format>
FORCE_INLINE uint32_t order_key(uint32_t bits) {
    if constexpr (format == DataFormat::Float16_b) {
        return argmax_bfloat16_key(bits);
    } else {
        return argmax_float32_key(bits);
    }
}

// Scans faces [face_begin, face_begin + face_count) of one tile and folds them into the running
// maxima of the rows they hold. A face sits at its own offset within the slot: (face - face_begin) faces in.
template <DataFormat format, uint32_t face_begin, uint32_t face_count>
void scan_tile(
    const InputContext& ctx,
    uint32_t slot_addr,
    uint32_t tile_x,
    uint32_t tile_y,
    uint32_t keys[],
    uint32_t arg_max[]) {
    using element_type = std::conditional_t<format == DataFormat::Float16_b, uint16_t, uint32_t>;
    for (uint32_t face_id = face_begin; face_id < face_begin + face_count; face_id++) {
        uint32_t rows_to_process = face_height;
        uint32_t cols_to_process = face_width;
        if (ctx.has_padding) {
            get_face_data_range(rows_to_process, cols_to_process, tile_x, tile_y, face_id, ctx);
        }
        if (rows_to_process == 0 && cols_to_process == 0) {
            continue;
        }
        // The read barrier has completed; the data no longer changes, so plain loads are safe.
        const element_type* face_ptr =
            reinterpret_cast<const element_type*>(slot_addr) + (face_id - face_begin) * face_size;
        const uint32_t row_base = ((face_id < 2) ? 0 : face_height) - (face_begin < 2 ? 0 : face_height);
        const uint32_t col_base = (tile_x * ctx.tile_width) + ((face_id == 1 || face_id == 3) ? face_width : 0);
        for (uint32_t row = 0; row < rows_to_process; row++) {
            uint32_t curr_key = keys[row_base + row];
            uint32_t curr_arg_max = arg_max[row_base + row];
            const element_type* row_ptr = face_ptr + row * face_width;
            for (uint32_t col = 0; col < cols_to_process; col++) {
                const uint32_t key = order_key<format>(row_ptr[col]);
                if (key > curr_key) {
                    curr_key = key;
                    curr_arg_max = col_base + col;
                }
            }
            keys[row_base + row] = curr_key;
            arg_max[row_base + row] = curr_arg_max;
        }
    }
}

}  // namespace

void kernel_main() {
    // Compile time args
    // -----------------
    constexpr auto src_page_size = get_arg(args::src_page_size);
    constexpr auto tile_height = get_arg(args::tile_height);
    constexpr auto tile_width = get_arg(args::tile_width);
    // Input padded size (last two dims) in tiles, and logical size in elements
    constexpr auto input_height = get_arg(args::input_height);
    constexpr auto input_width = get_arg(args::input_width);
    constexpr auto logical_height = get_arg(args::logical_height);
    constexpr auto logical_width = get_arg(args::logical_width);
    constexpr bool keepdim = static_cast<bool>(get_arg(args::keepdim));
    // Tiles per half of the double-buffered input DFB
    constexpr auto batch_tiles = get_arg(args::batch_tiles);
    // Faces this instance scans: 0-3 (alone), 0-1 (top half) or 2-3 (bottom half).
    constexpr auto face_begin = get_arg(args::face_begin);
    constexpr auto face_count = get_arg(args::face_count);
    constexpr auto face_bytes = get_arg(args::face_bytes);
    constexpr auto src_alignment = get_arg(args::src_alignment);
    constexpr auto cores_per_group = get_arg(args::cores_per_group);
    constexpr auto rows_per_group = get_arg(args::rows_per_group);
    // Byte stride of one index in the output staging area: 4 when one page holds a row of indices,
    // the output alignment when every index is its own page (keepdim).
    constexpr auto out_stride = get_arg(args::out_stride);

    // Runtime args
    // ------------
    const uint32_t row_begin = get_arg(args::row_begin);
    const uint32_t row_count = get_arg(args::row_count);
    const uint32_t col_begin = get_arg(args::col_begin);
    const uint32_t col_count = get_arg(args::col_count);
    const uint32_t core_in_group = get_arg(args::core_in_group);
    const uint32_t leader_x = get_arg(args::leader_x);
    const uint32_t leader_y = get_arg(args::leader_y);

    auto s_src = TensorAccessor(tensor::src);
    auto s_dst = TensorAccessor(tensor::dst);

    const Noc noc;
    const UnicastEndpoint remote;
    const DataflowBuffer src_dfb(dfb::src);
    const DataflowBuffer out_dfb(dfb::out);
    const DataflowBuffer partials_dfb(dfb::partials);
    Semaphore done_sem(sem::done);

    constexpr DataFormat src_data_format = get_dataformat(dfb::src);
    constexpr uint32_t element_size = src_data_format == DataFormat::Float16_b ? 2 : 4;
    const uint32_t default_key = order_key<src_data_format>(get_default_value<src_data_format>());

    const uint32_t src_base = src_dfb.get_write_ptr();
    const uint32_t out_base = out_dfb.get_write_ptr();
    const uint32_t partials_base = partials_dfb.get_write_ptr();

    // This instance's rows within a tile row, and the bytes of its faces in a slot.
    constexpr uint32_t row_offset = face_begin < 2 ? 0 : face_height;
    constexpr uint32_t rows_per_instance = face_count * face_height / 2;
    constexpr uint32_t slot_bytes = face_count * face_bytes;
    constexpr uint32_t tile_height_rem = logical_height % tile_height;
    constexpr uint32_t bottom_rows = tile_height_rem == 0 ? tile_height : tile_height_rem;
    const InputContext ctx(
        tile_height,
        tile_width,
        input_height,
        input_width,
        logical_height,
        logical_width,
        tile_height_rem,
        logical_width % tile_width,
        logical_height % face_height,
        logical_width % face_width,
        src_data_format,
        /*l1_dfb_addr=*/0);

    // Rows of tile row r that hold data and belong to this instance.
    auto rows_here = [&](uint32_t r) -> uint32_t {
        const uint32_t valid = (r % input_height == input_height - 1) ? bottom_rows : tile_height;
        if (valid <= row_offset) {
            return 0;
        }
        return valid - row_offset < rows_per_instance ? valid - row_offset : rows_per_instance;
    };

    // Partial results: keys, then indices, each laid out [core_in_group][row][rows_per_instance].
    constexpr uint32_t partial_row_bytes = rows_per_instance * sizeof(uint32_t);
    constexpr uint32_t partial_core_bytes = rows_per_group * partial_row_bytes;
    constexpr uint32_t partial_idx_words = cores_per_group * partial_core_bytes / sizeof(uint32_t);
    volatile tt_l1_ptr uint32_t* partials = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(partials_base);

    // Writes this instance's valid_rows indices of tile row r from the staging area.
    auto write_row = [&](uint32_t r, uint32_t valid_rows) {
        const uint32_t outer = r / input_height;
        const uint32_t first_row = ((r % input_height) * tile_height) + row_offset;
        if constexpr (keepdim) {
            for (uint32_t i = 0; i < valid_rows; i++) {
                noc.async_write(
                    out_dfb,
                    s_dst,
                    sizeof(uint32_t),
                    {.offset_bytes = i * out_stride},
                    {.page_id = (outer * logical_height) + first_row + i});
            }
        } else {
            noc.async_write(
                out_dfb,
                s_dst,
                valid_rows * sizeof(uint32_t),
                {.offset_bytes = 0},
                {.page_id = outer, .offset_bytes = first_row * static_cast<uint32_t>(sizeof(uint32_t))});
        }
        noc.async_write_barrier();
    };
    auto stage = [&](uint32_t i, uint32_t index) {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_base + i * out_stride)[0] = index;
    };

    uint32_t keys[rows_per_instance];
    uint32_t arg_max[rows_per_instance];
    auto reset_row = [&]() {
        for (uint32_t i = 0; i < rows_per_instance; i++) {
            keys[i] = default_key;
            arg_max[i] = 0;
        }
    };
    // Ends the scan of local row rr (tile row row_begin + rr).
    auto finish_row = [&](uint32_t rr) {
        const uint32_t r = row_begin + rr;
        const uint32_t valid_rows = rows_here(r);
        if constexpr (cores_per_group == 1) {
            if (valid_rows > 0) {
                for (uint32_t i = 0; i < valid_rows; i++) {
                    stage(i, arg_max[i]);
                }
                write_row(r, valid_rows);
            }
        } else {
            const uint32_t base = (core_in_group * partial_core_bytes + rr * partial_row_bytes) / sizeof(uint32_t);
            for (uint32_t i = 0; i < valid_rows; i++) {
                partials[base + i] = keys[i];
                partials[partial_idx_words + base + i] = arg_max[i];
            }
        }
    };

    // Reads item k of this core's (row, column) items into slot: this instance's faces, and of a
    // face only its rows that hold data (rounded up to the NoC alignment).
    const uint32_t num_items = row_count * col_count;
    auto issue_read = [&](uint32_t k, uint32_t slot) {
        const uint32_t r = row_begin + (k / col_count);
        const uint32_t rows = rows_here(r);
        if (rows == 0) {
            return;
        }
        const uint32_t tile_id = (r * input_width) + col_begin + (k % col_count);
        const uint32_t dst_offset = slot * slot_bytes;
        if (rows == rows_per_instance) {
            // Every face of this instance is full: one contiguous read.
            noc.async_read(
                s_src,
                src_dfb,
                slot_bytes,
                {.page_id = tile_id, .offset_bytes = face_begin * face_bytes},
                {.offset_bytes = dst_offset});
            return;
        }
        // Only the first rows hold data. With four faces, at most face_height rows means the bottom faces
        // are padding; read the data rows of the two top faces.
        const uint32_t face_rows = rows < face_height ? rows : face_height;
        uint32_t bytes = face_rows * face_width * element_size;
        bytes = (bytes + src_alignment - 1) / src_alignment * src_alignment;
        if (face_count == 4 && rows > face_height) {
            noc.async_read(s_src, src_dfb, src_page_size, {.page_id = tile_id}, {.offset_bytes = dst_offset});
            return;
        }
        for (uint32_t f = 0; f < 2; f++) {
            noc.async_read(
                s_src,
                src_dfb,
                bytes,
                {.page_id = tile_id, .offset_bytes = (face_begin + f) * face_bytes},
                {.offset_bytes = dst_offset + f * face_bytes});
        }
    };
    auto batch_end = [&](uint32_t b) {
        const uint32_t last = (b + 1) * batch_tiles;
        return last < num_items ? last : num_items;
    };
    auto issue_batch = [&](uint32_t b) {
        for (uint32_t k = b * batch_tiles; k < batch_end(b); k++) {
            issue_read(k, ((b & 1) * batch_tiles) + (k - b * batch_tiles));
        }
    };

    // Scan: double-buffered batches; at each barrier only the batch about to be scanned is in flight.
    const uint32_t num_batches = (num_items + batch_tiles - 1) / batch_tiles;
    reset_row();
    uint32_t current_rr = 0;
    if (num_batches > 0) {
        issue_batch(0);
    }
    for (uint32_t b = 0; b < num_batches; b++) {
        noc.async_read_barrier();
        // Keep the plain loads of scan_tile after the barrier.
        asm volatile("" ::: "memory");
        if (b + 1 < num_batches) {
            issue_batch(b + 1);
        }
        for (uint32_t k = b * batch_tiles; k < batch_end(b); k++) {
            const uint32_t rr = k / col_count;
            if (rr != current_rr) {
                finish_row(current_rr);
                reset_row();
                current_rr = rr;
            }
            const uint32_t r = row_begin + rr;
            if (rows_here(r) == 0) {
                continue;
            }
            const uint32_t slot = ((b & 1) * batch_tiles) + (k - b * batch_tiles);
            scan_tile<src_data_format, face_begin, face_count>(
                ctx, src_base + slot * slot_bytes, col_begin + (k % col_count), r % input_height, keys, arg_max);
        }
    }
    if (num_items > 0) {
        finish_row(current_rr);
    }

    if constexpr (cores_per_group > 1) {
        if (core_in_group != 0) {
            // Hand this core's partials to the leader, at the same offsets in its copy of the buffer.
            const uint32_t own_offset = core_in_group * partial_core_bytes;
            const uint32_t used_bytes = row_count * partial_row_bytes;
            noc.async_write(
                partials_dfb,
                remote,
                used_bytes,
                {.offset_bytes = own_offset},
                {.noc_x = leader_x, .noc_y = leader_y, .addr = partials_base + own_offset});
            noc.async_write(
                partials_dfb,
                remote,
                used_bytes,
                {.offset_bytes = (partial_idx_words * sizeof(uint32_t)) + own_offset},
                {.noc_x = leader_x,
                 .noc_y = leader_y,
                 .addr = partials_base + (partial_idx_words * sizeof(uint32_t)) + own_offset});
            noc.async_write_barrier();
            done_sem.up(noc, leader_x, leader_y, 1);
            noc.async_atomic_barrier();
        } else {
            done_sem.wait(cores_per_group - 1);
            // Every increment of this launch has arrived; reset for the next launch or trace replay.
            done_sem.set(0);
            for (uint32_t rr = 0; rr < row_count; rr++) {
                const uint32_t r = row_begin + rr;
                const uint32_t valid_rows = rows_here(r);
                const uint32_t row_word = (rr * partial_row_bytes) / sizeof(uint32_t);
                for (uint32_t i = 0; i < valid_rows; i++) {
                    // Column order: the first core's columns come first, and a later core replaces the
                    // running result only on a strictly greater key, as a single pass over the row would.
                    uint32_t best_key = partials[row_word + i];
                    uint32_t best_idx = partials[partial_idx_words + row_word + i];
                    for (uint32_t c = 1; c < cores_per_group; c++) {
                        const uint32_t word = (c * partial_core_bytes / sizeof(uint32_t)) + row_word + i;
                        const uint32_t key = partials[word];
                        if (key > best_key) {
                            best_key = key;
                            best_idx = partials[partial_idx_words + word];
                        }
                    }
                    stage(i, best_idx);
                }
                if (valid_rows > 0) {
                    write_row(r, valid_rows);
                }
            }
        }
    }
}

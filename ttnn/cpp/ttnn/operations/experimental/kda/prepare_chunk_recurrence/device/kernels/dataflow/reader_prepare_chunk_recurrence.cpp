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
    Noc noc;

    uint32_t valid_chunks;
    uint32_t partial_chunk;
    uint32_t partial_rows;
    {
        DataflowBuffer control(dfb::chronology_compute);
        DataflowBuffer writer_control(dfb::chronology_writer);
        control.reserve_back(1);
        writer_control.reserve_back(1);
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(control.get_write_ptr());
        auto* writer_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(writer_control.get_write_ptr());
        // Read both bounds together, each into its own channel's slot, so compute waits for one round trip.
        if constexpr (has_actual_start) {
            const auto start_tensor = TensorAccessor(*tensor::get_token_if_present<"actual_start">());
            noc.async_read(start_tensor, control, sizeof(uint32_t), {.page_id = 0}, {});
        }
        if constexpr (has_actual_end) {
            const auto end_tensor = TensorAccessor(*tensor::get_token_if_present<"actual_end">());
            noc.async_read(end_tensor, writer_control, sizeof(uint32_t), {.page_id = 0}, {});
        }
        if constexpr (has_actual_start || has_actual_end) {
            noc.async_read_barrier();
        }
        const uint32_t start = has_actual_start ? words[0] : 0;
        auto topology = kda_chronology::derive(start, sp_rank, sp_size, local_rows);
        if constexpr (has_actual_end) {
            topology = kda_chronology::derive_interval(start, writer_words[0], sp_rank, sp_size, local_rows);
        }
        valid_chunks = topology.chunk_count();
        partial_chunk = topology.partial_chunk();
        partial_rows = topology.partial_rows();
        kda_chronology::store(words, topology);
        control.push_back(1);
        kda_chronology::store(writer_words, topology);
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
        publish(q, Kt, false);
        read_batch(v_accessor, v, Vt);
        publish(v, Vt, true);
        read_batch(k_accessor, k, Kt);
        publish(k, Kt, true);
        done += batch_size;
    }

    // Once its reads are done this kernel writes the two FP32 outputs with the most tiles, v_beta and k_dec_t, on its
    // own NoC; the writer writes the rest on the other. Every core writes its outputs across all banks at about the
    // same time, so the writes are NoC-bound and splitting them over both NoCs shortens the tail after the last
    // batch. Compute publishes them per batch, as the writer drains them.
    const auto v_beta_accessor = TensorAccessor(tensor::v_beta_output);
    const auto k_decay_transposed_accessor = TensorAccessor(tensor::k_decay_transposed_output);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    const auto drain = [&](auto& buffer, const auto& accessor, uint32_t tiles) {
        buffer.wait_front(batch_size * tiles);
        for (uint32_t item = 0; item < batch_size; ++item) {
            for (uint32_t tile = 0; tile < tiles; ++tile) {
                noc.async_write(
                    buffer,
                    accessor,
                    buffer.get_entry_size(),
                    {.offset_bytes = (item * tiles + tile) * buffer.get_entry_size()},
                    {.page_id = batch_work_items[item] * tiles + tile});
            }
        }
        noc.async_write_barrier();
        buffer.pop_front(batch_size * tiles);
    };
    next = 0;
    for (uint32_t done = 0; done < valid_items;) {
        batch_size = valid_items - done >= batch_items ? batch_items : 1;
        for (uint32_t item = 0; item < batch_size; ++next) {
            if ((work_item_start + next) % num_chunks < valid_chunks) {
                batch_work_items[item++] = work_item_start + next;
            }
        }
        drain(v_beta, v_beta_accessor, Ct * Vt);
        drain(k_decay_transposed, k_decay_transposed_accessor, Kt * Ct);
        done += batch_size;
    }
}

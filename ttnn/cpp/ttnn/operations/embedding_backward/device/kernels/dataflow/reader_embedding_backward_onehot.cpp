// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Reader for the output-stationary embedding backward. Work units u = r * num_hidden_chunks + c (output tile row r
// of the weight gradient, hidden chunk c) are dealt round-robin, so the chunks of a frequently hit row land on
// different cores. Each core loads every index, buckets the grad tile rows g that hit each output tile row, and for
// each (r, c, g) pushes a one-hot tile P (P[i][j] = 1 when token j of grad tile row g has index r * 32 + i) and the
// grad tiles of chunk c. The compute kernel accumulates P x grad into DEST, so every output tile is written exactly
// once.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

constexpr uint32_t TILE_DIM = 32;
constexpr uint32_t FACE_DIM = 16;
constexpr uint32_t NOT_SEEN = 0xFFFFFFFF;
constexpr uint32_t PAIR_BATCH = 4;
constexpr uint32_t P_SLOTS = 2 * PAIR_BATCH;

void kernel_main() {
    const uint32_t grad_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t first_unit = get_arg_val<uint32_t>(2);
    const uint32_t unit_stride = get_arg_val<uint32_t>(3);
    const uint32_t total_units = get_arg_val<uint32_t>(4);

    constexpr uint32_t hidden_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t chunk_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t num_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t num_index_pages = get_compile_time_arg_val(3);
    constexpr uint32_t index_page_len = get_compile_time_arg_val(4);
    constexpr bool index_is_bf16 = get_compile_time_arg_val(5) == 1;
    constexpr uint32_t num_tokens = num_index_pages * index_page_len;
    constexpr uint32_t num_grad_rows = num_tokens / TILE_DIM;

    constexpr auto grad_args = TensorAccessorArgs<6>();
    constexpr auto index_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    const auto grad_s = TensorAccessor(grad_args, grad_tensor_addr);
    const auto index_s = TensorAccessor(index_args, index_tensor_addr);

    constexpr uint32_t cb_grad_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_onehot_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_meta_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_scratch_idx = tt::CBIndex::c_25;

    Noc noc;
    CircularBuffer cb_grad(cb_grad_idx);
    CircularBuffer cb_onehot(cb_onehot_idx);
    CircularBuffer cb_meta(cb_meta_idx);
    CircularBuffer cb_scratch(cb_scratch_idx);

    const uint32_t num_rows = total_units / num_chunks;

    // Scratch layout: indices (uint32 x num_tokens), pair list (uint16 x num_tokens), then three uint32 arrays of
    // num_rows: count, offset and last seen grad row.
    const uint32_t scratch = cb_scratch.get_write_ptr();
    auto* idx = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch);
    const uint32_t list_addr = scratch + num_tokens * sizeof(uint32_t);
    auto* list = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(list_addr);
    const uint32_t arrays_addr = (list_addr + num_tokens * sizeof(uint16_t) + 15) & ~15u;
    auto* count = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(arrays_addr);
    auto* offset = count + num_rows;
    auto* last = offset + num_rows;

    // Load every index. BFLOAT16 indices land in the pair-list area first (2 bytes each, the same size) and are
    // decoded into the uint32 array.
    constexpr uint32_t index_elem_bytes = index_is_bf16 ? 2 : 4;
    const uint32_t load_addr = index_is_bf16 ? list_addr : scratch;
    for (uint32_t b = 0; b < num_index_pages; ++b) {
        noc.async_read(
            index_s,
            CoreLocalMem<uint32_t>(load_addr + b * index_page_len * index_elem_bytes),
            index_page_len * index_elem_bytes,
            {.page_id = b},
            {});
    }
    noc.async_read_barrier();
    if constexpr (index_is_bf16) {
        for (uint32_t t = 0; t < num_tokens; ++t) {
            union {
                float f;
                uint32_t u;
            } v;
            v.u = static_cast<uint32_t>(list[t]) << 16;
            idx[t] = static_cast<uint32_t>(v.f);
        }
    }

    // Count, per output tile row, the distinct grad tile rows that hit it.
    for (uint32_t k = 0; k < num_rows; ++k) {
        count[k] = 0;
        last[k] = NOT_SEEN;
    }
    for (uint32_t g = 0; g < num_grad_rows; ++g) {
        for (uint32_t j = 0; j < TILE_DIM; ++j) {
            const uint32_t k = idx[g * TILE_DIM + j] / TILE_DIM;
            if (k < num_rows && last[k] != g) {
                last[k] = g;
                ++count[k];
            }
        }
    }
    uint32_t running = 0;
    for (uint32_t k = 0; k < num_rows; ++k) {
        offset[k] = running;
        running += count[k];
        count[k] = 0;
        last[k] = NOT_SEEN;
    }
    for (uint32_t g = 0; g < num_grad_rows; ++g) {
        for (uint32_t j = 0; j < TILE_DIM; ++j) {
            const uint32_t k = idx[g * TILE_DIM + j] / TILE_DIM;
            if (k < num_rows && last[k] != g) {
                last[k] = g;
                list[offset[k] + count[k]] = g;
                ++count[k];
            }
        }
    }

    // One-hot tiles are built in place. Each slot remembers which entries it set, so reuse only clears those.
    // Up to PAIR_BATCH pairs are read under one barrier. The one-hot ring holds P_SLOTS tiles and the grad ring
    // P_SLOTS windows of chunk_tiles, both advance one entry per pair, so each pair's slot is addressed from the ring
    // bases and each pair is still pushed as its own window.
    const uint32_t onehot_tile_bytes = get_tile_size(cb_onehot_idx);
    const uint32_t grad_tile_bytes = get_tile_size(cb_grad_idx);
    const uint32_t onehot_base = cb_onehot.get_write_ptr();
    const uint32_t grad_base = cb_grad.get_write_ptr();
    uint16_t set_pos[P_SLOTS][TILE_DIM];
    uint32_t set_count[P_SLOTS] = {0};
    bool slot_clean[P_SLOTS] = {false};
    uint32_t slot = 0;

    for (uint32_t u = first_unit; u < total_units; u += unit_stride) {
        const uint32_t r = u / num_chunks;
        const uint32_t c = u % num_chunks;
        const uint32_t first_tile = c * chunk_tiles;
        constexpr uint32_t tiles = chunk_tiles;
        const uint32_t n = count[r];

        cb_meta.reserve_back(1);
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(cb_meta.get_write_ptr())[0] = n;
        cb_meta.push_back(1);

        for (uint32_t m0 = 0; m0 < n; m0 += PAIR_BATCH) {
            const uint32_t batch = (n - m0 < PAIR_BATCH) ? n - m0 : PAIR_BATCH;
            cb_onehot.reserve_back(batch);
            cb_grad.reserve_back(batch * tiles);
            for (uint32_t q = 0; q < batch; ++q) {
                const uint32_t g = list[offset[r] + m0 + q];
                const uint32_t sl = (slot + q) % P_SLOTS;

                auto* p = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(onehot_base + sl * onehot_tile_bytes);
                if (!slot_clean[sl]) {
                    auto* w = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(p);
                    for (uint32_t z = 0; z < onehot_tile_bytes / sizeof(uint32_t); ++z) {
                        w[z] = 0;
                    }
                    slot_clean[sl] = true;
                } else {
                    for (uint32_t z = 0; z < set_count[sl]; ++z) {
                        p[set_pos[sl][z]] = 0;
                    }
                }
                set_count[sl] = 0;
                for (uint32_t j = 0; j < TILE_DIM; ++j) {
                    const uint32_t index = idx[g * TILE_DIM + j];
                    if (index / TILE_DIM == r) {
                        const uint32_t i = index % TILE_DIM;
                        const uint32_t pos = (((i / FACE_DIM) * 2 + (j / FACE_DIM)) * FACE_DIM * FACE_DIM) +
                                             ((i % FACE_DIM) * FACE_DIM) + (j % FACE_DIM);
                        p[pos] = 0x3F80;  // bfloat16 1.0
                        set_pos[sl][set_count[sl]++] = pos;
                    }
                }

                uint32_t grad_l1 = grad_base + sl * tiles * grad_tile_bytes;
                for (uint32_t t = 0; t < tiles; ++t) {
                    noc.async_read(
                        grad_s,
                        CoreLocalMem<uint32_t>(grad_l1),
                        grad_tile_bytes,
                        {.page_id = g * hidden_tiles + first_tile + t},
                        {});
                    grad_l1 += grad_tile_bytes;
                }
            }
            noc.async_read_barrier();
            for (uint32_t q = 0; q < batch; ++q) {
                cb_onehot.push_back(1);
                cb_grad.push_back(tiles);
            }
            slot = (slot + batch) % P_SLOTS;
        }
    }
}

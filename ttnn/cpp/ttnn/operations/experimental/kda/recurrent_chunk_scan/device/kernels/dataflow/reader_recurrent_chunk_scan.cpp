// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// KDA scan reader: the initial state S [K,V] once, then vector-decay prep
// intermediates v_beta, kd, q_decay, intra, k_dec_t, dl[K,1], t_inv. FP32 by default; selected intermediates may be
// BF16.

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include <tt-metalium/constants.hpp>
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

template <typename Accessor>
FORCE_INLINE void read_and_publish_contiguous_tiles(
    const Accessor& accessor, DataflowBuffer& buffer, Noc& noc, uint32_t base, uint32_t count) {
    buffer.reserve_back(count);
    const uint32_t entry_size = buffer.get_entry_size();
    uint32_t tile = 0;
    for (const auto& page : accessor.pages(base, base + count)) {
        noc.async_read(accessor, buffer, entry_size, {.page_id = page.page_id()}, {.offset_bytes = tile * entry_size});
        ++tile;
    }
    noc.async_read_barrier();
    // push_back publishes this DFB to compute. Complete its reads first; delaying publication to coalesce across
    // buffers would prevent compute from overlapping the next buffer's NOC reads.
    buffer.push_back(count);
}

template <uint32_t Vt, uint32_t VtFull, typename Accessor>
FORCE_INLINE void read_and_publish_value_slice(
    const Accessor& accessor,
    DataflowBuffer& buffer,
    Noc& noc,
    uint32_t row_base,
    uint32_t rows,
    uint32_t value_block) {
    buffer.reserve_back(rows * Vt);
    const uint32_t entry_size = buffer.get_entry_size();
    for (uint32_t row = 0; row < rows; ++row) {
        const uint32_t source = row_base + row * VtFull + value_block * Vt;
        const uint32_t destination = row * Vt;
        for (uint32_t value_tile = 0; value_tile < Vt; ++value_tile) {
            noc.async_read(
                accessor,
                buffer,
                entry_size,
                {.page_id = source + value_tile},
                {.offset_bytes = (destination + value_tile) * entry_size});
        }
    }
    noc.async_read_barrier();
    // push_back publishes this DFB to compute. Complete its reads first; delaying publication to coalesce across
    // buffers would prevent compute from overlapping the next buffer's NOC reads.
    buffer.push_back(rows * Vt);
}

template <uint32_t Kt, uint32_t Vt>
FORCE_INLINE void seed_identity(DataflowBuffer& buffer, Noc& noc, uint32_t value_block) {
    constexpr uint32_t one_fp32 = __builtin_bit_cast(uint32_t, 1.0F);
    constexpr uint32_t face_elements = tt::constants::FACE_HW;
    constexpr uint32_t tile_elements = tt::constants::TILE_HW;
    constexpr uint32_t tile_count = Kt * Vt;

    buffer.reserve_back(tile_count);
    noc.async_write_zeros(buffer, tile_count * buffer.get_entry_size());
    noc.write_zeros_l1_barrier();
    {
        auto lock = buffer.scoped_write_lock(tile_count);
        auto state_ptr = lock.get_ptr<volatile uint32_t>();
        for (uint32_t local_col = 0; local_col < Vt; ++local_col) {
            const uint32_t global_col = value_block * Vt + local_col;
            if (global_col < Kt) {
                auto tile = state_ptr + (global_col * Vt + local_col) * tile_elements;
                for (uint32_t row = 0; row < tt::constants::FACE_HEIGHT; ++row) {
                    tile[row * tt::constants::FACE_WIDTH + row] = one_fp32;
                    tile[3 * face_elements + row * tt::constants::FACE_WIDTH + row] = one_fp32;
                }
            }
        }
    }
    buffer.push_back(tile_count);
}

template <uint32_t Ct, uint32_t Kt, uint32_t Vt, uint32_t Vt_full, uint32_t summary_pair>
TT_KERNEL void reader(uint32_t head, uint32_t value_block, uint32_t num_chunks) {
    const auto v_beta_accessor = TensorAccessor(tensor::v_beta);
    const auto kd_accessor = TensorAccessor(tensor::kd);
    const auto k_decay_transposed_accessor = TensorAccessor(tensor::k_decay_transposed);
    const auto final_decay_accessor = TensorAccessor(tensor::final_decay);
    const auto t_inv_accessor = TensorAccessor(tensor::t_inv);

    DataflowBuffer state(dfb::state);
    DataflowBuffer t_inv(dfb::t_inv);
    DataflowBuffer v_beta(dfb::v_beta);
    DataflowBuffer kd(dfb::kd);
    DataflowBuffer q_decay(dfb::q_decay);
    DataflowBuffer intra(dfb::intra);
    DataflowBuffer summary_seed(dfb::summary_seed);
    DataflowBuffer k_decay_transposed(dfb::k_decay_transposed);
    DataflowBuffer final_decay(dfb::final_decay);
    Noc noc;

    constexpr uint32_t chunk_chunk_tiles = Ct * Ct;
    constexpr uint32_t chunk_key_tiles = Ct * Kt;
    constexpr uint32_t key_chunk_tiles = Kt * Ct;
    constexpr uint32_t key_value_tiles = Kt * Vt;

    if constexpr (summary_pair) {
        state.reserve_back(key_value_tiles);
        noc.async_write_zeros(state, key_value_tiles * state.get_entry_size());
        noc.write_zeros_l1_barrier();
        state.push_back(key_value_tiles);
        seed_identity<Kt, Vt>(summary_seed, noc, value_block);
    } else {
        const auto initial_state_accessor = TensorAccessor(tensor::initial_state);
        read_and_publish_value_slice<Vt, Vt_full>(
            initial_state_accessor, state, noc, head * Kt * Vt_full, Kt, value_block);
    }

    for (uint32_t chunk = 0; chunk < num_chunks; ++chunk) {
        const uint32_t head_chunk = head * num_chunks + chunk;
        if constexpr (summary_pair) {
            read_and_publish_contiguous_tiles(kd_accessor, kd, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
            read_and_publish_value_slice<Vt, Vt_full>(
                v_beta_accessor, v_beta, noc, head_chunk * Ct * Vt_full, Ct, value_block);
            read_and_publish_contiguous_tiles(
                t_inv_accessor, t_inv, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
            read_and_publish_contiguous_tiles(
                k_decay_transposed_accessor, k_decay_transposed, noc, head_chunk * key_chunk_tiles, key_chunk_tiles);
            read_and_publish_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt);
        } else {
            read_and_publish_value_slice<Vt, Vt_full>(
                v_beta_accessor, v_beta, noc, head_chunk * Ct * Vt_full, Ct, value_block);
            read_and_publish_contiguous_tiles(kd_accessor, kd, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
            const auto q_decay_accessor = TensorAccessor(tensor::q_decay);
            const auto intra_accessor = TensorAccessor(tensor::intra);
            read_and_publish_contiguous_tiles(
                q_decay_accessor, q_decay, noc, head_chunk * chunk_key_tiles, chunk_key_tiles);
            read_and_publish_contiguous_tiles(
                intra_accessor, intra, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
            read_and_publish_contiguous_tiles(
                k_decay_transposed_accessor, k_decay_transposed, noc, head_chunk * key_chunk_tiles, key_chunk_tiles);
            read_and_publish_contiguous_tiles(final_decay_accessor, final_decay, noc, head_chunk * Kt, Kt);
            read_and_publish_contiguous_tiles(
                t_inv_accessor, t_inv, noc, head_chunk * chunk_chunk_tiles, chunk_chunk_tiles);
        }
    }
}

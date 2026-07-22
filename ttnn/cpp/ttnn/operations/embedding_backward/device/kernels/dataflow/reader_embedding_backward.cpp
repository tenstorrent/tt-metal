// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/core_local_mem.h"

constexpr uint32_t INPUT_SIZE = 32;

FORCE_INLINE uint32_t get_index(uint32_t input_l1_addr, uint32_t idx) {
    constexpr bool is_index_bfloat16 = get_compile_time_arg_val(5) == 1;
    if constexpr (is_index_bfloat16) {
        auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(input_l1_addr);
        union {
            float f;
            uint32_t u;
        } u;
        u.u = (uint32_t)input_l1_ptr[idx] << 16;
        return static_cast<uint32_t>(u.f);
    } else {
        auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);
        return input_l1_ptr[idx];
    }
}

// TODO: Helper for printing mask (can remove this)
FORCE_INLINE uint32_t get_mask(uint32_t input_l1_addr, uint32_t idx) {
    auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(input_l1_addr);
    return input_l1_ptr[idx];
}

FORCE_INLINE uint32_t process_index_chunk(uint32_t index_l1_addr, uint32_t chunk_indexes[INPUT_SIZE]) {
    uint32_t chunk_count = 0;

    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        uint32_t idx = get_index(index_l1_addr, i);
        uint32_t chunk_id = idx >> 5;  // equivalent to idx / 32

        bool is_new_chunk = true;
        for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
            if (chunk_indexes[chunk] == chunk_id) {
                is_new_chunk = false;
                break;
            }
        }

        if (is_new_chunk) {
            chunk_indexes[chunk_count++] = chunk_id;
        }
    }

    return chunk_count;
}

FORCE_INLINE void generate_mask(uint32_t index_l1_addr, uint32_t chunk_id, uint32_t mask_l1_addr) {
    auto mask_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t*>(mask_l1_addr);

    uint32_t x_min = chunk_id << 5;  // equivalent to chunk_id * 32
    uint32_t x_max = x_min + INPUT_SIZE;

    for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
        uint32_t idx = get_index(index_l1_addr, i);
        uint8_t mask = ~static_cast<uint8_t>(0);  // equivalent to numeric_limits<uint8_t>::max()
        if (idx >= x_min && idx < x_max) {
            mask = idx & (INPUT_SIZE - 1);  // equivalent to idx % INPUT_SIZE
        }
        mask_l1_ptr[i] = mask;
    }
}

FORCE_INLINE void generate_zeros_cb(uint32_t input_l1_addr) {
    constexpr bool is_output_bfloat16 = get_compile_time_arg_val(6) == 1;
    auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(input_l1_addr);

    if constexpr (is_output_bfloat16) {
        // 512 * 4 = 2048 bytes = single tile of bfloat16
        for (uint32_t i = 0; i < 512; ++i) {
            input_l1_ptr[i] = 0;
        }
    } else {
        // 272 * 4 = 1088 bytes = single tile of bfloat8_b
        for (uint32_t i = 0; i < 272; ++i) {
            input_l1_ptr[i] = 0;
        }
    }
}

void kernel_main() {
    const uint32_t grad_tensor_addr = get_arg_val<uint32_t>(0);
    const uint32_t index_tensor_addr = get_arg_val<uint32_t>(1);
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t tiles_per_hidden = get_arg_val<uint32_t>(3);
    const uint32_t hidden_offset = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(5);

    constexpr uint32_t max_tiles_per_core = get_compile_time_arg_val(0);
    constexpr uint32_t batch_size = get_compile_time_arg_val(1);
    constexpr uint32_t seq_len_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t num_embeddings = get_compile_time_arg_val(3);

    constexpr uint32_t cb_grad_idx = tt::CBIndex::c_0;
    constexpr uint32_t cb_index_idx = tt::CBIndex::c_1;
    constexpr uint32_t cb_out_intermed_idx = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_idx = tt::CBIndex::c_24;
    constexpr uint32_t cb_chunk_count_scratch_idx = tt::CBIndex::c_25;
    constexpr uint32_t cb_id_out0_idx = tt::CBIndex::c_16;

    constexpr uint32_t grad_page_size = get_tile_size(cb_grad_idx);
    constexpr uint32_t out_page_size = get_tile_size(cb_id_out0_idx);

    constexpr auto grad_args = TensorAccessorArgs<7>();
    constexpr auto index_args = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<index_args.next_compile_time_args_offset()>();

    const auto grad_s = TensorAccessor(grad_args, grad_tensor_addr);
    const auto index_s = TensorAccessor(index_args, index_tensor_addr);
    const auto out_s = TensorAccessor(out_args, output_tensor_addr);

    Noc noc;
    CircularBuffer cb_grad(cb_grad_idx);
    CircularBuffer cb_index(cb_index_idx);
    CircularBuffer cb_out_intermed(cb_out_intermed_idx);
    CircularBuffer cb_mask(cb_mask_idx);
    CircularBuffer cb_chunk_count_scratch(cb_chunk_count_scratch_idx);
    CircularBuffer cb_id_out0(cb_id_out0_idx);

    uint32_t index_block_size = get_tile_size(cb_index_idx) >> 5;  // we only need 32 elements
    uint32_t index_l1_addr = cb_index.get_write_ptr();             // static

    uint32_t chunk_indexes[INPUT_SIZE];

    // ZERO out output this core is responsible for
    uint32_t out_read_ptr = cb_out_intermed.get_read_ptr();
    // Fill one tile with zeros
    generate_zeros_cb(out_read_ptr);
    // Fill all of output (for this core) to zeros
    uint32_t out_tile_idx = hidden_offset;
    for (uint32_t i = 0; i < num_embeddings; ++i) {
        for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
            noc.async_write(
                CoreLocalMem<uint32_t>(out_read_ptr), out_s, out_page_size, {}, {.page_id = out_tile_idx + hidden_dim});
        }
        out_tile_idx += tiles_per_hidden;
    }
    noc.async_write_barrier();

    uint32_t chunk_count_l1_addr = cb_chunk_count_scratch.get_read_ptr();
    auto chunk_count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(chunk_count_l1_addr);

    uint32_t grad_tile_idx = hidden_offset;
    for (uint32_t b = 0; b < batch_size; ++b) {
        uint32_t index_seq_offset = 0;
        for (uint32_t s = 0; s < seq_len_tiles; ++s) {
            noc.async_read(
                index_s,
                CoreLocalMem<uint32_t>(index_l1_addr),
                index_block_size,
                {.page_id = b, .offset_bytes = index_seq_offset},
                {});
            noc.async_read_barrier();

            index_seq_offset += index_block_size;

            // maps the next chunk of indexes to the corresponding output masks
            uint32_t chunk_count = process_index_chunk(index_l1_addr, chunk_indexes);

            // Pass chunk_count to compute UNPACK
            chunk_count_ptr[0] = chunk_count;

            cb_grad.reserve_back(max_tiles_per_core);
            uint32_t grad_write_ptr = cb_grad.get_write_ptr();
            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                noc.async_read(
                    grad_s,
                    CoreLocalMem<uint32_t>(grad_write_ptr),
                    grad_page_size,
                    {.page_id = grad_tile_idx + hidden_dim},
                    {});
                grad_write_ptr += grad_page_size;
            }
            noc.async_read_barrier();
            cb_grad.push_back(max_tiles_per_core);
            grad_tile_idx += tiles_per_hidden;

            for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
                uint32_t chunk_idx = chunk_indexes[chunk];

                cb_mask.reserve_back(1);
                uint32_t mask_l1_addr = cb_mask.get_write_ptr();
                generate_mask(index_l1_addr, chunk_idx, mask_l1_addr);

// TODO: Remove debug prints
#if 0
                for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
                    uint32_t idx = get_index(index_l1_addr, i);
                    uint32_t msk = get_mask(mask_l1_addr, i);
                    DPRINT("{}: {} -> {}\n", chunk, idx, msk);
                }
#endif
                cb_mask.push_back(1);

                cb_out_intermed.reserve_back(max_tiles_per_core);
                uint32_t out_write_ptr = cb_out_intermed.get_write_ptr();
                out_tile_idx = chunk_idx * tiles_per_hidden + hidden_offset;
                for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                    noc.async_read(
                        out_s,
                        CoreLocalMem<uint32_t>(out_write_ptr),
                        out_page_size,
                        {.page_id = out_tile_idx + hidden_dim},
                        {});
                    out_write_ptr += out_page_size;
                }
                noc.async_read_barrier();
                cb_out_intermed.push_back(max_tiles_per_core);

                cb_id_out0.wait_front(max_tiles_per_core);
                uint32_t out0_read_ptr = cb_id_out0.get_read_ptr();
                for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                    noc.async_write(
                        CoreLocalMem<uint32_t>(out0_read_ptr),
                        out_s,
                        out_page_size,
                        {},
                        {.page_id = out_tile_idx + hidden_dim});
                    out0_read_ptr += out_page_size;
                }
                noc.async_write_barrier();
                cb_id_out0.pop_front(max_tiles_per_core);
            }  // chunk_count
        }  // seq_len_tiles
    }  // batch_size
}

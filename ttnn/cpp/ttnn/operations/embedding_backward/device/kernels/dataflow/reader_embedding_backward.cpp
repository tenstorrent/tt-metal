// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

constexpr uint32_t INPUT_SIZE = 32;

FORCE_INLINE uint64_t get_index_noc_address(uint32_t tile_idx, uint32_t offset = 0) {
    const std::uint32_t index_tensor_addr = get_arg_val<uint32_t>(1);
    constexpr bool index_stick_size_is_power_of_two = get_compile_time_arg_val(4) == 1;
    constexpr bool index_is_dram = get_compile_time_arg_val(1) == 1;

    if constexpr (index_stick_size_is_power_of_two) {
        constexpr uint32_t index_log2_stick_size = get_compile_time_arg_val(5);
        InterleavedPow2AddrGen<index_is_dram> index = {.bank_base_address = index_tensor_addr,
                                                       .log_base_2_of_page_size = index_log2_stick_size};
        return get_noc_addr(tile_idx, index, offset);
    } else {
        constexpr uint32_t index_page_size = get_compile_time_arg_val(3);
        InterleavedAddrGen<index_is_dram> index = {.bank_base_address = index_tensor_addr,
                                                   .page_size = index_page_size};
        return get_noc_addr(tile_idx, index, offset);
    }
}

FORCE_INLINE uint32_t get_index(uint32_t input_l1_addr, uint32_t idx) {
    constexpr bool is_index_bfloat16 = get_compile_time_arg_val(6) == 1;
    if constexpr (is_index_bfloat16) {
        auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t *>(input_l1_addr);
        union {
            float f;
            uint32_t u;
        } u;
        u.u = (uint32_t)input_l1_ptr[idx] << 16;
        return static_cast<uint32_t>(u.f);
    } else {
        auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(input_l1_addr);
        return input_l1_ptr[idx];
    }
}

// TODO: Helper for printing mask (can remove this)
FORCE_INLINE uint32_t get_mask(uint32_t input_l1_addr, uint32_t idx) {
    auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(input_l1_addr);
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
    auto mask_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint8_t *>(mask_l1_addr);

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
    constexpr bool is_output_bfloat16 = get_compile_time_arg_val(7) == 1;
    auto input_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(input_l1_addr);

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
    const uint32_t output_tensor_addr = get_arg_val<uint32_t>(2);
    const uint32_t tiles_per_hidden = get_arg_val<uint32_t>(3);
    const uint32_t hidden_offset = get_arg_val<uint32_t>(4);
    const uint32_t tiles_per_core = get_arg_val<uint32_t>(5);

    constexpr bool grad_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr bool out_is_dram = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t max_tiles_per_core = get_compile_time_arg_val(8);
    constexpr uint32_t batch_size = get_compile_time_arg_val(9);
    constexpr uint32_t seq_len_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t num_embeddings = get_compile_time_arg_val(11);

    constexpr uint32_t cb_grad = tt::CB::c_in0;
    constexpr uint32_t cb_index = tt::CB::c_in1;
    constexpr uint32_t cb_out_intermed = tt::CB::c_in2;
    constexpr uint32_t cb_mask = tt::CB::c_intermed0;
    constexpr uint32_t cb_chunk_count_scratch = tt::CB::c_intermed1;
    constexpr uint32_t cb_id_out0 = tt::CB::c_out0;

    constexpr uint32_t grad_page_size = get_tile_size(cb_grad);
    constexpr uint32_t out_page_size = get_tile_size(cb_id_out0);
    constexpr DataFormat grad_data_format = get_dataformat(cb_grad);
    constexpr DataFormat out_data_format = get_dataformat(cb_id_out0);

    const InterleavedAddrGenFast<grad_is_dram> grad_s = {
        .bank_base_address = grad_tensor_addr, .page_size = grad_page_size, .data_format = grad_data_format};

    const InterleavedAddrGenFast<out_is_dram> out_s = {
        .bank_base_address = output_tensor_addr, .page_size = out_page_size, .data_format = out_data_format};

    uint32_t index_block_size = get_tile_size(cb_index) >> 5;  // we only need 32 elements
    uint32_t index_l1_addr = get_write_ptr(cb_index);          // static

    uint32_t chunk_indexes[INPUT_SIZE];

    // ZERO out output this core is responsible for
    uint32_t out_read_ptr = get_read_ptr(cb_out_intermed);
    // Fill one tile with zeros
    generate_zeros_cb(out_read_ptr);
    // Fill all of output (for this core) to zeros
    uint32_t out_tile_idx = hidden_offset;
    for (uint32_t i = 0; i < num_embeddings; ++i) {
        for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
            noc_async_write_tile(out_tile_idx + hidden_dim, out_s, out_read_ptr);
        }
        out_tile_idx += tiles_per_hidden;
    }
    noc_async_write_barrier();

    uint32_t chunk_count_l1_addr = get_read_ptr(cb_chunk_count_scratch);
    auto chunk_count_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(chunk_count_l1_addr);

    uint32_t grad_tile_idx = hidden_offset;
    for (uint32_t b = 0; b < batch_size; ++b) {
        uint64_t index_seq_noc_addr = get_index_noc_address(b);
        for (uint32_t s = 0; s < seq_len_tiles; ++s) {
            noc_async_read(index_seq_noc_addr, index_l1_addr, index_block_size);
            noc_async_read_barrier();

            index_seq_noc_addr += index_block_size;

            // maps the next chunk of indexes to the corresponding output masks
            uint32_t chunk_count = process_index_chunk(index_l1_addr, chunk_indexes);

            // Pass chunk_count to compute UNPACK
            chunk_count_ptr[0] = chunk_count;

            cb_reserve_back(cb_grad, max_tiles_per_core);
            uint32_t grad_write_ptr = get_write_ptr(cb_grad);
            for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                noc_async_read_tile(grad_tile_idx + hidden_dim, grad_s, grad_write_ptr);
                grad_write_ptr += grad_page_size;
            }
            noc_async_read_barrier();
            cb_push_back(cb_grad, max_tiles_per_core);
            grad_tile_idx += tiles_per_hidden;

            for (uint32_t chunk = 0; chunk < chunk_count; ++chunk) {
                uint32_t chunk_idx = chunk_indexes[chunk];

                cb_reserve_back(cb_mask, 1);
                uint32_t mask_l1_addr = get_write_ptr(cb_mask);
                generate_mask(index_l1_addr, chunk_idx, mask_l1_addr);

// TODO: Remove debug prints
#if 0
                for (uint32_t i = 0; i < INPUT_SIZE; ++i) {
                    uint32_t idx = get_index(index_l1_addr, i);
                    uint32_t msk = get_mask(mask_l1_addr, i);
                    DPRINT << chunk << ": " << idx << " -> " << msk << ENDL();
                }
#endif
                cb_push_back(cb_mask, 1);

                cb_reserve_back(cb_out_intermed, max_tiles_per_core);
                uint32_t out_write_ptr = get_write_ptr(cb_out_intermed);
                out_tile_idx = chunk_idx * tiles_per_hidden + hidden_offset;
                for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                    noc_async_read_tile(out_tile_idx + hidden_dim, out_s, out_write_ptr);
                    out_write_ptr += out_page_size;
                }
                noc_async_read_barrier();
                cb_push_back(cb_out_intermed, max_tiles_per_core);

                cb_wait_front(cb_id_out0, max_tiles_per_core);
                uint32_t out_read_ptr = get_read_ptr(cb_id_out0);
                for (uint32_t hidden_dim = 0; hidden_dim < tiles_per_core; hidden_dim++) {
                    noc_async_write_tile(out_tile_idx + hidden_dim, out_s, out_read_ptr);
                    out_read_ptr += out_page_size;
                }
                noc_async_write_barrier();
                cb_pop_front(cb_id_out0, max_tiles_per_core);
            }  // chunk_count
        }  // seq_len_tiles
    }  // batch_size
}

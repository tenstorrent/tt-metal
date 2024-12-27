// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include <vector>

#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/dataflow/dataflow_common.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/transformer/speculative_sdpa_decode/device/kernels/speculative_common.hpp"

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "debug/assert.h"

void kernel_main() {
    /*
    In DRAM, Q is (B, PNHt, DHt), K is (B, St, DHt), V is (B, St, DHt), mask is (B, PNHt, PSt)
    We want to read for a particular batch cur_batch, and sequence length up to padded layer length.
    We read Q: (cur_batch, PNHt, DHt), K: (cur_batch, PSt, DHt), V: (cur_batch, PSt, DHt), mask: (cur_batch, PNHt, PSt)
    */
    constexpr uint32_t B = get_compile_time_arg_val(0);           // batch size
    constexpr uint32_t PNHt = get_compile_time_arg_val(1);        // padded number of heads in tiles
    constexpr uint32_t St = get_compile_time_arg_val(2);          // full sequence length of kv cache in tiles
    constexpr uint32_t DHt = get_compile_time_arg_val(3);         // head dim
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);  // number of tiles in seqlen of a k/v/mask chunk
    constexpr uint32_t num_cores = get_compile_time_arg_val(5);
    constexpr bool is_q_sharded = get_compile_time_arg_val(6);
    constexpr uint32_t num_cores_per_batch = get_compile_time_arg_val(7);
    constexpr uint32_t k_chunk_size = get_compile_time_arg_val(8);
    constexpr uint32_t index_stick_size_B = get_compile_time_arg_val(9);
    constexpr bool is_paged_attention = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t num_kv_heads = get_compile_time_arg_val(11);
    constexpr uint32_t block_size_t = get_compile_time_arg_val(12);
    constexpr uint32_t Bkv = get_compile_time_arg_val(13);
    constexpr uint32_t num_cores_per_head = get_compile_time_arg_val(14);
    constexpr uint32_t num_heads_per_core = get_compile_time_arg_val(15);
    constexpr uint32_t num_output_cores = get_compile_time_arg_val(16);
    constexpr bool is_causal = get_compile_time_arg_val(17) == 1;
    constexpr bool use_attention_mask = get_compile_time_arg_val(18) == 1;

    constexpr uint32_t Spec_chunk_t =
        get_compile_time_arg_val(19);  // speculative chunk size (in tiles), for the first and last chunk
    constexpr uint32_t speculative_chunk_size = Spec_chunk_t * tt::constants::TILE_HEIGHT;
    uint32_t output_semaphore_addr = get_semaphore(get_compile_time_arg_val(20));  // semaphore for output ready
    constexpr bool is_output_sharded = get_compile_time_arg_val(21) == 1;

    uint32_t arg_idx = 0;
    const uint32_t q_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t k_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t v_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t pos_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mask_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_spec_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_table_page_size = get_arg_val<uint32_t>(arg_idx++);
    const bool is_worker = get_arg_val<uint32_t>(arg_idx++) == 0;
    const bool is_output_core = get_arg_val<uint32_t>(arg_idx++) == 1;
    const uint32_t cur_head_group = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_batch = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_reduce = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t core_num_in_output = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t cur_pos_arg = get_arg_val<uint32_t>(arg_idx++);

    // idle core
    if (q_addr == 0) {
        return;
    }
    // Get cur_pos
    constexpr uint32_t cur_pos_base = St * 32 - 1;
    uint32_t cur_pos = cur_pos_base;  // default to non-causal, which we do attention on the entire kv cache. In this
                                      // case we set cur_pos to the last position
    if constexpr (is_causal) {
        // using UINT32_MAX as a flag to indicate that cur_pos is not provided as a list
        if (cur_pos_arg != UINT32_MAX) {
            cur_pos = cur_pos_arg;
        } else {
            constexpr uint32_t cb_index_id = tt::CBIndex::c_8;
            const InterleavedAddrGen<true> addrg = {.bank_base_address = pos_addr, .page_size = index_stick_size_B};

            cb_reserve_back(cb_index_id, 1);
            uint32_t index_cb_wr_ptr = get_write_ptr(cb_index_id);
            // index_tensor has one page to read
            uint64_t tensor_index_noc_addr = get_noc_addr(0, addrg);
            noc_async_read(tensor_index_noc_addr, index_cb_wr_ptr, index_stick_size_B);
            noc_async_read_barrier();
            cb_push_back(cb_index_id, 1);
            volatile tt_l1_ptr uint32_t* index_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(index_cb_wr_ptr);
            cur_pos = index_ptr[cur_batch];
        }

        if (cur_pos == UINT32_MAX) {
            // cur_pos of -1 indicates that the user should be skipped
            return;
        }
    }

    volatile tt_l1_ptr uint32_t* page_table_ptr;
    if constexpr (is_paged_attention) {
        constexpr uint32_t cb_id_page_table = tt::CBIndex::c_9;
        const InterleavedAddrGen<true> page_table_gen = {
            .bank_base_address = page_table_addr, .page_size = page_table_page_size};
        cb_reserve_back(cb_id_page_table, 1);
        uint32_t page_table_cb_wr_ptr = get_write_ptr(cb_id_page_table);
        uint64_t page_table_noc_addr = get_noc_addr(cur_batch, page_table_gen);
        noc_async_read(page_table_noc_addr, page_table_cb_wr_ptr, page_table_page_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_page_table, 1);
        page_table_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(page_table_cb_wr_ptr);
    }
    // Sequence length assignment
    auto
        [k_num_chunks,
         k_chunk_start,
         k_chunk_end,
         speculative_height_dim_start_tile_offset1,
         speculative_height_dim_start_tile_offset2,
         non_spec_height_dim_start_tile_offset,
         adjusted_cur_pos_for_non_spec,
         adjusted_cur_pos_for_spec,
         do_speculative_compute] =
            get_speculative_runtime_args(
                cur_pos,
                cur_batch,
                core_num_in_reduce,
                num_cores_per_head,
                k_chunk_size,
                speculative_chunk_size,
                Spec_chunk_t);
    tt_l1_ptr uint32_t* all_output_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_output_cores;
    tt_l1_ptr uint32_t* all_output_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx++));

    DPRINT << "k_num_chunks: " << k_num_chunks << ENDL();
    DPRINT << "k_chunk_start: " << k_chunk_start << ENDL();
    DPRINT << "k_chunk_end: " << k_chunk_end << ENDL();
    DPRINT << "spec_h_dim_t_offset1: " << speculative_height_dim_start_tile_offset1 << ENDL();
    DPRINT << "spec_h_dim_t_offset2: " << speculative_height_dim_start_tile_offset2 << ENDL();
    DPRINT << "non_spec_h_dim_t_offset: " << non_spec_height_dim_start_tile_offset << ENDL();
    DPRINT << "adj_cur_pos_non_spec: " << adjusted_cur_pos_for_non_spec << ENDL();
    DPRINT << "adj_cur_pos_spec: " << adjusted_cur_pos_for_spec << ENDL();
    DPRINT << "do_spec_comp: " << (uint32_t)do_speculative_compute << ENDL();

    uint32_t output_core_noc_x = all_output_noc_x[cur_batch];
    uint32_t output_core_noc_y = all_output_noc_y[cur_batch];

    if (k_chunk_start == k_chunk_end) {
        return;  // early exit because no computes needs to be done
    }

    constexpr uint32_t q_chunk_tiles = PNHt * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t spec_chunk_tiles = Spec_chunk_t * DHt;
    constexpr uint32_t mask_chunk_tiles = PNHt * Sk_chunk_t;
    constexpr uint32_t spec_mask_chunk_tiles = PNHt * Spec_chunk_t;

    constexpr bool is_dram = true;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;

    constexpr uint32_t onetile = 1;
    constexpr uint32_t q_tile_bytes = get_tile_size(cb_q_in);
    constexpr DataFormat q_data_format = get_dataformat(cb_q_in);
    constexpr uint32_t k_tile_bytes = get_tile_size(cb_k_in);
    constexpr DataFormat k_data_format = get_dataformat(cb_k_in);
    constexpr uint32_t v_tile_bytes = get_tile_size(cb_v_in);
    constexpr DataFormat v_data_format = get_dataformat(cb_v_in);
    constexpr uint32_t mask_tile_bytes = get_tile_size(cb_mask_in);
    constexpr DataFormat mask_data_format = get_dataformat(cb_mask_in);

    constexpr uint32_t barrier_threshold = get_barrier_read_threshold<q_tile_bytes, num_cores>();
    uint32_t barrier_count = 0;

    // First, read Q entirely, it could be interleaved or sharded
    uint32_t q_batch_offset = cur_batch * q_chunk_tiles;
    uint32_t q_chunk_tiles_bytes = q_chunk_tiles * q_tile_bytes;

    if constexpr (is_q_sharded) {
        uint64_t q_read_addr;
        if (is_output_core) {
            q_read_addr = get_noc_addr(q_addr);
        } else {
            q_read_addr = get_noc_addr(output_core_noc_x, output_core_noc_y, q_addr);
        }
        cb_reserve_back(cb_q_in, q_chunk_tiles);
        uint32_t q_write_ptr = get_write_ptr(cb_q_in);
        noc_async_read(q_read_addr, q_write_ptr, q_chunk_tiles_bytes);
        noc_async_read_barrier();
        cb_push_back(cb_q_in, q_chunk_tiles);
    } else {
        const InterleavedAddrGenFast<is_dram> q_reader = {
            .bank_base_address = q_addr, .page_size = q_tile_bytes, .data_format = q_data_format};
        uint32_t q_tile_id = q_batch_offset;
        cb_reserve_back(cb_q_in, q_chunk_tiles);
        uint32_t q_write_ptr = get_write_ptr(cb_q_in);
        for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
            noc_async_read_tile(q_tile_id, q_reader, q_write_ptr);
            q_tile_id += 1;
            q_write_ptr += q_tile_bytes;
            if (++barrier_count == barrier_threshold) {
                noc_async_read_barrier();
                barrier_count = 0;
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_q_in, q_chunk_tiles);
    }

    // Read the rest
    const InterleavedAddrGenFast<is_dram> k_reader = {
        .bank_base_address = k_addr, .page_size = k_tile_bytes, .data_format = k_data_format};

    const InterleavedAddrGenFast<is_dram> v_reader = {
        .bank_base_address = v_addr, .page_size = v_tile_bytes, .data_format = v_data_format};

    const InterleavedAddrGenFast<is_dram> mask_reader = {
        .bank_base_address = mask_addr, .page_size = mask_tile_bytes, .data_format = mask_data_format};

    // Read Speculative chunks to speculate
    if (do_speculative_compute) {
        uint32_t speculative_start_offset1 = speculative_height_dim_start_tile_offset1 *
                                             DHt;  // the first speculative chunk reads from the start of the seqlen
        uint32_t speculative_start_offset2 = speculative_height_dim_start_tile_offset2 * DHt;
        for (uint32_t cur_head = cur_head_group * num_heads_per_core;
             cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
             ++cur_head) {
            // Batch and head level offset
            const uint32_t mask_batch_offset = (cur_batch % Bkv) * PNHt * St;
            const uint32_t k_batch_offset = (cur_batch % Bkv) * num_kv_heads * St * DHt;
            const uint32_t v_batch_offset = (cur_batch % Bkv) * num_kv_heads * St * DHt;
            const uint32_t k_head_offset = cur_head * St * DHt;
            const uint32_t v_head_offset = cur_head * St * DHt;

            // Read first speculative chunk
            {
                //// Compute offset for mask
                const uint32_t mask_chunk_offset = speculative_height_dim_start_tile_offset1 * Spec_chunk_t;
                uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;

                //// Compute offset for K and V
                const uint32_t k_chunk_offset = speculative_start_offset1;
                const uint32_t v_chunk_offset = speculative_start_offset1;
                uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;
                uint32_t v_start_tile_id = v_batch_offset + v_head_offset + v_chunk_offset;

                read_kv_mask_chunks<
                    DHt,
                    Spec_chunk_t,
                    barrier_threshold,
                    spec_chunk_tiles,
                    spec_mask_chunk_tiles,
                    mask_tile_bytes,
                    PNHt,
                    use_attention_mask,
                    cb_k_in,
                    cb_v_in,
                    cb_mask_in>(
                    0,  // read one chunk only
                    1,  // read one chunk only
                    k_start_tile_id,
                    v_start_tile_id,
                    mask_start_tile_id,
                    k_reader,
                    v_reader,
                    mask_reader,
                    k_tile_bytes,
                    v_tile_bytes,
                    St);
            }

            // Read last speculative chunk
            {
                //// Compute offset for mask
                const uint32_t mask_chunk_offset = speculative_height_dim_start_tile_offset2 * Spec_chunk_t;
                uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;

                //// Compute offset for K and V
                const uint32_t k_chunk_offset = speculative_start_offset2;
                const uint32_t v_chunk_offset = speculative_start_offset2;
                uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset;
                uint32_t v_start_tile_id = v_batch_offset + v_head_offset + v_chunk_offset;

                read_kv_mask_chunks<
                    DHt,
                    Spec_chunk_t,
                    barrier_threshold,
                    spec_chunk_tiles,
                    spec_mask_chunk_tiles,
                    mask_tile_bytes,
                    PNHt,
                    use_attention_mask,
                    cb_k_in,
                    cb_v_in,
                    cb_mask_in>(
                    0,  // read one chunk only
                    1,  // read one chunk only
                    k_start_tile_id,
                    v_start_tile_id,
                    mask_start_tile_id,
                    k_reader,
                    v_reader,
                    mask_reader,
                    k_tile_bytes,
                    v_tile_bytes,
                    St);
            }
        }
    }

    // Read Non-speculative chunks to compute
    uint32_t non_speculative_start_offset = non_spec_height_dim_start_tile_offset * DHt;
    for (uint32_t cur_head = cur_head_group * num_heads_per_core;
         cur_head < cur_head_group * num_heads_per_core + num_heads_per_core;
         ++cur_head) {
        const uint32_t mask_batch_offset = (cur_batch % Bkv) * PNHt * St;
        const uint32_t mask_chunk_offset = k_chunk_start * Sk_chunk_t;
        uint32_t mask_start_tile_id = mask_batch_offset + mask_chunk_offset;
        if constexpr (is_paged_attention) {
            for (uint32_t k_chunk = k_chunk_start; k_chunk < k_chunk_end; ++k_chunk) {
                // Read K chunk in row-major order (to simplify page mapping). Write tiles to CB in transposed order.
                const uint32_t k_chunk_start_row_num = k_chunk * Sk_chunk_t;
                cb_reserve_back(cb_k_in, k_chunk_tiles);
                uint32_t k_write_ptr = get_write_ptr(cb_k_in);
                barrier_count = 0;
                for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                    uint32_t k_write_ptr_col = k_write_ptr + row * k_tile_bytes;
                    uint32_t virtual_k_tile_row_num = k_chunk_start_row_num + row;
                    uint32_t physical_k_tile_id =
                        virtual_seq_tile_id_to_physical_tile_id<num_kv_heads, block_size_t, DHt>(
                            virtual_k_tile_row_num, cur_head, page_table_ptr);
                    for (uint32_t col = 0; col < DHt; ++col) {
                        noc_async_read_tile(physical_k_tile_id, k_reader, k_write_ptr_col);
                        physical_k_tile_id += 1;                       // Go to next tile in row
                        k_write_ptr_col += Sk_chunk_t * k_tile_bytes;  // Go to next column in CB

                        if (++barrier_count == barrier_threshold) {
                            noc_async_read_barrier();
                            barrier_count = 0;
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_k_in, k_chunk_tiles);

                if constexpr (use_attention_mask) {
                    mask_start_tile_id = read_mask_chunk<
                        cb_mask_in,
                        mask_chunk_tiles,
                        mask_tile_bytes,
                        barrier_threshold,
                        PNHt,
                        Sk_chunk_t>(St, mask_start_tile_id, mask_reader);
                }

                // Read V chunk in row major order, write in row-major order
                cb_reserve_back(cb_v_in, k_chunk_tiles);
                uint32_t v_write_ptr = get_write_ptr(cb_v_in);
                barrier_count = 0;

                for (uint32_t row = 0; row < Sk_chunk_t; ++row) {
                    uint32_t virtual_v_tile_row_num = k_chunk_start_row_num + row;
                    uint32_t physical_v_tile_id =
                        virtual_seq_tile_id_to_physical_tile_id<num_kv_heads, block_size_t, DHt>(
                            virtual_v_tile_row_num, cur_head, page_table_ptr);
                    for (uint32_t col = 0; col < DHt; ++col) {
                        noc_async_read_tile(physical_v_tile_id, v_reader, v_write_ptr);
                        physical_v_tile_id += 1;
                        v_write_ptr += v_tile_bytes;

                        if (++barrier_count == barrier_threshold) {
                            noc_async_read_barrier();
                            barrier_count = 0;
                        }
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_v_in, k_chunk_tiles);
            }

        } else {
            // Offset for current batch
            const uint32_t k_batch_offset = (cur_batch % Bkv) * num_kv_heads * St * DHt;
            const uint32_t v_batch_offset = (cur_batch % Bkv) * num_kv_heads * St * DHt;
            const uint32_t k_head_offset = cur_head * St * DHt;
            const uint32_t v_head_offset = cur_head * St * DHt;

            // Then, read K, V, Mask k_chunk_tiles at a time
            const uint32_t k_chunk_offset = k_chunk_start * Sk_chunk_t * DHt;
            const uint32_t v_chunk_offset = k_chunk_start * Sk_chunk_t * DHt;
            uint32_t k_start_tile_id = k_batch_offset + k_head_offset + k_chunk_offset + non_speculative_start_offset;
            uint32_t v_start_tile_id = v_batch_offset + v_head_offset + v_chunk_offset + non_speculative_start_offset;

            read_kv_mask_chunks<
                DHt,
                Sk_chunk_t,
                barrier_threshold,
                k_chunk_tiles,
                mask_chunk_tiles,
                mask_tile_bytes,
                PNHt,
                use_attention_mask,
                cb_k_in,
                cb_v_in,
                cb_mask_in>(
                k_chunk_start,
                k_chunk_end,
                k_start_tile_id,
                v_start_tile_id,
                mask_start_tile_id,
                k_reader,
                v_reader,
                mask_reader,
                k_tile_bytes,
                v_tile_bytes,
                St);
        }
    }

    // if do verification, we wait for semaphore to know that ground truth output and speculative output are written to
    // output tensor memory assume output core also does verification
    if (is_output_core) {
        volatile tt_l1_ptr uint32_t* output_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(output_semaphore_addr);
        // num signals to wait is the 2*num_reducer_cores (1 signal for speculative and 1 for ground truth per head)
        DPRINT << "waiting for output done semaphore \n";
        noc_semaphore_wait(output_semaphore_addr_ptr, 2 * (num_kv_heads / num_heads_per_core));
        DPRINT << "got output done semaphore with value " << *output_semaphore_addr_ptr << "\n";

        // ground truth output and speculative output have the same shape and format as input q
        if constexpr (is_output_sharded) {
            ASSERT(false);  // TODO: implement sharded output
        } else {
            // read ground truth output
            {
                const InterleavedAddrGenFast<is_dram> out_reader = {
                    .bank_base_address = out_addr, .page_size = q_tile_bytes, .data_format = q_data_format};
                uint32_t q_tile_id = q_batch_offset;
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                uint32_t q_write_ptr = get_write_ptr(cb_q_in);
                for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
                    noc_async_read_tile(q_tile_id, out_reader, q_write_ptr);
                    q_tile_id += 1;
                    q_write_ptr += q_tile_bytes;
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_q_in, q_chunk_tiles);
            }

            // read speculative output
            {
                const InterleavedAddrGenFast<is_dram> out_spec_reader = {
                    .bank_base_address = out_spec_addr, .page_size = q_tile_bytes, .data_format = q_data_format};
                uint32_t q_tile_id = q_batch_offset;
                cb_reserve_back(cb_q_in, q_chunk_tiles);
                uint32_t q_write_ptr = get_write_ptr(cb_q_in);
                for (uint32_t tile = 0; tile < q_chunk_tiles; ++tile) {
                    noc_async_read_tile(q_tile_id, out_spec_reader, q_write_ptr);
                    q_tile_id += 1;
                    q_write_ptr += q_tile_bytes;
                    if (++barrier_count == barrier_threshold) {
                        noc_async_read_barrier();
                        barrier_count = 0;
                    }
                }
                noc_async_read_barrier();
                cb_push_back(cb_q_in, q_chunk_tiles);
            }
        }
    }
}

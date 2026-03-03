// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

template <uint32_t val_size>
FORCE_INLINE void fill_with_val(uint32_t start_addr, uint32_t n_bytes, uint32_t val) {
    static_assert(val_size == 2 || val_size == 4, "Unsupported val_size");
    using IntType = std::conditional_t<(val_size == 2), uint16_t, uint32_t>;

    uint32_t end_addr = start_addr + n_bytes;
    uint32_t start_addr_4B = (start_addr + 0x3) & 0xFFFFFFFC;
    uint32_t end_addr_4B = end_addr & 0xFFFFFFFC;

    // 4B writes
    auto* start_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(start_addr_4B);
    auto* end_ptr_4B = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(end_addr_4B);
    for (auto* p = start_ptr_4B; p < end_ptr_4B; ++p) {
        *p = val;
    }

    if constexpr (val_size < 4) {
        auto* start_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr);
        auto* end_ptr = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr);
        auto* start_ptr_a = reinterpret_cast<volatile tt_l1_ptr IntType*>(start_addr_4B);
        auto* end_ptr_a = reinterpret_cast<volatile tt_l1_ptr IntType*>(end_addr_4B);
        IntType val_ = static_cast<IntType>(val);

        for (auto* p = start_ptr; p < start_ptr_a; ++p) {
            *p = val_;
        }
        for (auto* p = end_ptr_a; p < end_ptr; ++p) {
            *p = val_;
        }
    }
}

void kernel_main() {
    // CT
    constexpr uint32_t cb_id_0 = get_compile_time_arg_val(0);
    constexpr uint32_t element_size = get_compile_time_arg_val(1);
    constexpr uint32_t tile_h = get_compile_time_arg_val(2);
    constexpr uint32_t tile_w = get_compile_time_arg_val(3);

    // RT
    uint32_t rt = 0;
    const uint32_t src_base_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t logical_width = get_arg_val<uint32_t>(rt++);  // this core's logical width (in elements)
    const uint32_t padded_width = get_arg_val<uint32_t>(rt++);
    const uint32_t logical_height_core = get_arg_val<uint32_t>(rt++);  // this core's real height (rows)
    const uint32_t padded_height_core = get_arg_val<uint32_t>(rt++);    // this core's output height (rows)
    const uint32_t global_logical_height = get_arg_val<uint32_t>(rt++); // full tensor logical height (rows) for batch stride
    const uint32_t shard_start_row = get_arg_val<uint32_t>(rt++);   // This core's start row (within a batch)
    const uint32_t start_col_bytes = get_arg_val<uint32_t>(rt++);   // for later block support.
    const uint32_t tiles_per_row = get_arg_val<uint32_t>(rt++);
    const uint32_t tile_rows_core = get_arg_val<uint32_t>(rt++);
    const uint32_t num_batches  = get_arg_val<uint32_t>(rt++);
    const uint32_t packed_pad_value = get_arg_val<uint32_t>(rt++);

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10)>;

    constexpr uint32_t page_size_jump = get_compile_time_arg_val(6);
    constexpr uint32_t pages_per_tensor_row = get_compile_time_arg_val(7);

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = src_base_addr, .shard_array = mapping_table};
#endif

    const uint32_t row_bytes = logical_width * element_size;
    const uint32_t padded_row_bytes = padded_width * element_size;
    const uint32_t width_pad_bytes = padded_row_bytes  - row_bytes;

    // We generate exactly this core's shard output (tile_row_core * tile_h rows, but clamp/pad via logical/padded heights)
    for (uint32_t b = 0; b < num_batches; ++b) {
        const uint32_t batch_row_base = b * global_logical_height;

        for (uint32_t tr = 0; tr < tile_rows_core) {
            cb_reserve_back(cb_id_0, tiles_per_row);
            uint32_t l1 = get_write_ptr(cb_id_0);

            // 32 rows inside a tile-row
            for (uint32_t r_in = 0; r_in < tile_h; ++r_in) {
                const uint32_t local_row = tr * tile_h + r_in;
                const bool is_real_row = local_row < logical_height_core;

                if (is_real_row) {
                    const uint32_t global_row = batch_row_base + shard_start_row + local_row;
                    const uint32_t row_page_base = global_row * pages_per_tensor_row;
                    const uint32_t page_index_offset = start_col_bytes / page_size_jump;
                    uint32_t offset_in_page = start_col_bytes - (page_index_offset * page_size_jump);
                    uint32_t page_id = row_page_base + page_index_offset;
                    uint32_t remaining = row_bytes;

                    // Read the real row segment across pages.
                    while (remaining > 0) {
                        const uint32_t max_bytes_this_page = page_size_jump - offset_in_page;
                        const uint32_t read_size = remaining < max_bytes_this_page ? remaining : max_bytes_this_page;
                        const uint64_t src_row_addr = get_noc_addr(page_id, s0, offset_in_page);
                        noc_async_read(src_row_addr, l1, read_size);
                        l1 += read_size;
                        remaining -= read_size;
                        page_id++;
                        offset_in_page = 0;
                    }

                    // Width padding (if any)
                    if (width_pad_bytes > 0) {
                        if constexpr (element_size == 2) {
                            fill_with_val<2>(l1, width_pad_bytes, packed_pad_value);
                        } else {
                            fill_with_val<4>(l1, width_pad_bytes, packed_pad_value);
                        }
                        l1 += width_pad_bytes;
                    }
                } else {
                    // Height padding row: full padded row = pad value
                    if constexpr (element_size == 2) {
                        fill_with_val<2>(l1, padded_row_bytes, packed_pad_value);
                    } else {
                        fill_with_val<4>(l1, padded_row_bytes, packed_pad_value);
                    }
                    l1 += padded_row_bytes;
                }
            }
            // One barrier per tile-row
            noc_async_read_barrier();
            cb_push_back(cb_id_0, tiles_per_row);
        }
    }
}

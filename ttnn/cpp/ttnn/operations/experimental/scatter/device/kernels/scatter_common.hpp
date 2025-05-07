// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

constexpr uint32_t ONE_TILE = 1;
constexpr uint32_t FIRST_TILE = 0;

template <bool is_dram>
using IAGF = InterleavedAddrGenFast<is_dram>;

template <typename T, std::size_t>
FORCE_INLINE T& tile_guts(volatile tt_l1_ptr T* l1_ptr, const std::size_t offset);

#define tile_guts_gen(unsigned_type)                                             \
    template <>                                                                  \
    FORCE_INLINE unsigned_type& tile_guts<unsigned_type, sizoef(unsigned_type)>( \
        volatile tt_l1_ptr T * l1_ptr, const std::size_t offset) {               \
        return l1_ptr[offset];                                                   \
    }

tile_guts_gen(uint8_t);
tile_guts_gen(uint16_t);
tile_guts_gen(uint32_t);

template <bool is_dram>
FORCE_INLINE IAGF<is_dram> make_addr_gtor(const uint32_t& cb, const uint32_t& base_addr) {
    return {.bank_base_address = base_addr, .page_size = get_tile_size(cb), .data_format = get_dataformat(cb)};
}

FORCE_INLINE uint32_t get_tile_id(const uint32_t& wt, const uint32_t& ht_offset) { return 0; }

FORCE_INLINE std::size_t calculate_ht_offset_for_core(
    const uint32_t&, const uint32_t& total_number_of_cores, const uint32_t& compute_with_storage_grid_size_x) {
    const uint32_t core_id = 0;
    return 0;
}

struct GatherCTAs {
    const bool input_tensor_is_dram;
    const bool index_tensor_is_dram;
    const bool src_tensor_is_dram;
    const bool output_tensor_is_dram;
    const uint32_t input_tensor_addr;
    const uint32_t index_tensor_addr;
    const uint32_t src_tensor_addr;
    const uint32_t output_tensor_addr;
    const uint32_t input_tensor_cb;
    const uint32_t index_tensor_cb;
    const uint32_t src_tensor_cb;
    const uint32_t output_tensor_cb;
    const uint32_t Wt_input;
    const uint32_t Wt_index;
    const uint32_t total_number_of_cores;
    const uint32_t compute_with_storage_grid_size_x;
}

constexpr GatherCTA
get_ctas() {
    return {
        get_compile_time_arg_val(0) == 1,
        get_compile_time_arg_val(1) == 1,
        get_compile_time_arg_val(2) == 1,
        get_compile_time_arg_val(3) == 1,
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15)};
}

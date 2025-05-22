// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr uint32_t cb0_id = get_compile_time_arg_val(2);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(3);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(4);
constexpr bool last_dim = get_compile_time_arg_val(5);
constexpr uint32_t num_banks = get_compile_time_arg_val(6);
constexpr bool use_best_effort = get_compile_time_arg_val(7);

template <bool DRAM>
inline void pack_full_contig(uint32_t contig_total, uint32_t& tile_id, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t total_local = 0;
    while (total_local < contig_total) {
        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        uint32_t id = tile_id;
        for (uint32_t j = 0; j < packet_size_in_pages; j++) {
            noc_async_read_tile(id, addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            id += num_banks;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
        tile_id++;
        total_local++;
        if (total_local % num_banks == 0) {
            tile_id += num_banks * (packet_size_in_pages - 1);
        }
    }
}

template <bool DRAM>
inline void pack_2contig_bf8(uint32_t contig_total, uint32_t& tile_id, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t total_local = 0;
    while (total_local < contig_total) {
        cb_reserve_back(cb0_id, 2);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        uint32_t id = tile_id;
        for (uint32_t j = 0; j < 2; j++) {
            noc_async_read_tile(id, addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            id += num_banks;
        }
        tile_id++;
        total_local++;
        noc_async_read_barrier();
        cb_push_back(cb0_id, 2);
        if (total_local % num_banks == 0) {
            tile_id += num_banks;
        }
    }
}

template <bool DRAM>
inline void pack_non_contig(uint32_t num_tiles, uint32_t& tile_id, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t total_local = 0;
    while (total_local < num_tiles) {
        cb_reserve_back(cb0_id, 1);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        uint32_t id = tile_id;
        for (uint32_t j = 0; j < 1; j++) {
            noc_async_read_tile(id, addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            id += num_banks;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
        tile_id++;
        total_local++;
    }
}

template <bool DRAM>
inline void pack_dim3_bf16_remain_even(
    uint32_t num_tiles, uint32_t tile_id, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    const uint32_t num_contig2 = (tile_cols_per_chip / (num_banks * packet_size_in_pages)) * num_banks;
    const uint32_t num_contig1 = ((tile_cols_per_chip - num_contig2 * 2) / num_banks) * num_banks;
    const uint32_t num_orphan = tile_cols_per_chip - num_contig2 * 2 - num_contig1;
    const uint32_t row = num_tiles / tile_cols_per_chip;
    for (uint32_t i = 0; i < row; i++) {
        pack_full_contig(num_contig2, tile_id, addrgen);
        pack_non_contig(num_contig1, tile_id, addrgen);
        pack_non_contig(num_orphan, tile_id, addrgen);
    }
}

template <bool DRAM>
inline void pack_dim3_bf8_dram_remain048(
    uint32_t num_tiles, uint32_t tile_id, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t row = num_tiles / tile_cols_per_chip;
    uint32_t num_full_contig = (tile_cols_per_chip / (num_banks * packet_size_in_pages)) * num_banks;
    uint32_t num_contig2 =
        ((tile_cols_per_chip - num_full_contig * packet_size_in_pages) / (num_banks * 2)) * num_banks;
    uint32_t num_contig1 = 0;
    uint32_t num_orphan = tile_cols_per_chip - num_full_contig * packet_size_in_pages - num_contig2 * 2;
    for (uint32_t i = 0; i < row; i++) {
        pack_full_contig(num_full_contig, tile_id, addrgen);
        pack_2contig_bf8(num_contig2, tile_id, addrgen);
        pack_non_contig(num_orphan, tile_id, addrgen);
    }
}

template <bool DRAM>
inline void pack_dim2_bf8(
    uint32_t filled_bank_tiles,
    uint32_t rest_full_contig_ids,
    uint32_t& tile_id,
    uint32_t rest_tiles,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    uint32_t rest_half_contig_ids, rest_orphan_tiles;
    bool skip_num_banks = false;
    if (num_banks * 3 < rest_tiles) {
        rest_half_contig_ids = (num_banks - rest_full_contig_ids);
        rest_orphan_tiles = rest_half_contig_ids;
        skip_num_banks = true;
    } else if (num_banks * 2 <= rest_tiles) {
        rest_half_contig_ids = num_banks;
        rest_orphan_tiles = (rest_tiles) % (num_banks * 2);
    } else if (num_banks < rest_tiles) {
        rest_half_contig_ids = (rest_tiles) % num_banks;
        rest_orphan_tiles = num_banks - rest_half_contig_ids;
    } else {
        rest_half_contig_ids = 0;
        rest_orphan_tiles = rest_tiles;
    }

    pack_2contig_bf8(rest_half_contig_ids, tile_id, tensor0_addrgen);
    if (skip_num_banks) {
        tile_id += 2 * num_banks - rest_half_contig_ids;
    }
    pack_non_contig(rest_orphan_tiles, tile_id, tensor0_addrgen);
}

template <bool DRAM>
inline void pack_dim2_bf16(
    uint32_t num_tiles_per_chip,
    uint32_t filled_bank_rows,
    uint32_t& tile_id,
    uint32_t rest_tiles,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    auto rest_orphan_tiles = 0;
    if (num_banks * 1 < rest_tiles) {
        rest_orphan_tiles = num_banks - (rest_tiles % (num_banks * (packet_size_in_pages - 1)));
    } else {
        rest_orphan_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    }
    pack_non_contig(rest_orphan_tiles, tile_id, tensor0_addrgen);
}

template <bool DRAM>
inline void pack_dim2(uint32_t num_tiles_per_chip, uint32_t tile_id, InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    auto filled_bank_rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
    auto rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    auto filled_bank_tiles = filled_bank_rows * num_banks * packet_size_in_pages;
    auto rest_full_contig_ids = 0;
    if (num_banks * (packet_size_in_pages - 1) < rest_tiles) {
        rest_full_contig_ids = rest_tiles % (num_banks * (packet_size_in_pages - 1));
    }
    pack_full_contig(filled_bank_rows * num_banks + rest_full_contig_ids, tile_id, tensor0_addrgen);
    if constexpr (packet_size_in_pages == 2) {
        pack_dim2_bf16<DRAM>(num_tiles_per_chip, filled_bank_rows, tile_id, rest_tiles, tensor0_addrgen);
    } else {
        pack_dim2_bf8<DRAM>(filled_bank_tiles, rest_full_contig_ids, tile_id, rest_tiles, tensor0_addrgen);
    }
}

template <bool DRAM>
inline void pack_generic(uint32_t num_tiles, uint32_t tile_id, InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    for (uint32_t i = 0; i < num_tiles; i += packet_size_in_pages) {
        uint32_t num_pages_to_read = min(num_tiles - i, packet_size_in_pages);
        cb_reserve_back(cb0_id, num_pages_to_read);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            noc_async_read_tile(tile_id, tensor0_addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, num_pages_to_read);
    }
}

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_chip = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_cols_per_chip = get_arg_val<uint32_t>(arg_idx++);

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    uint32_t tile_id = tile_id_start;
    if constexpr (use_best_effort) {
        if constexpr (last_dim) {
            if constexpr (packet_size_in_pages == 2) {
                pack_dim3_bf16_remain_even(num_tiles_per_chip, tile_id, tile_cols_per_chip, tensor0_addrgen);
            } else {
                pack_dim3_bf8_dram_remain048<is_dram>(num_tiles_per_chip, tile_id, tile_cols_per_chip, tensor0_addrgen);
            }
        } else {
            pack_dim2(num_tiles_per_chip, tile_id, tensor0_addrgen);
        }
    } else {
        pack_generic<is_dram>(num_tiles_per_chip, tile_id, tensor0_addrgen);
    }
}

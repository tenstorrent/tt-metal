// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
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

template <bool DRAM>
inline void pack_non_contig_tiles(uint32_t num_tiles, uint32_t& tile_id, InterleavedAddrGenFast<DRAM>& addrgen) {
    DPRINT << "[R][" << (uint32_t)my_chip_id << "] pack_non_contig_tiles tile_id: " << tile_id
           << ", num_tiles: " << num_tiles << "\n";
    for (uint32_t i = 0; i < num_tiles; i += packet_size_in_pages) {
        uint32_t num_pages_to_read = min(num_tiles - i, packet_size_in_pages);
        cb_reserve_back(cb0_id, num_pages_to_read);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] i: " << i << ", j: " << j << ", tile_id: " << tile_id
                   << "\n";
            noc_async_read_tile(tile_id, addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            tile_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, num_pages_to_read);
    }
}

inline uint32_t dim3_rel2abs_tile_id(
    uint32_t rel_tile_id, uint32_t tile_cols_per_chip, uint32_t ring_size, uint32_t ring_idx) {
    uint32_t row = rel_tile_id / tile_cols_per_chip;
    uint32_t idx = rel_tile_id % tile_cols_per_chip;
    return idx + (tile_cols_per_chip * ring_idx) + row * ring_size * tile_cols_per_chip;
}

inline uint32_t dim3_abs2rel_tile_id(
    uint32_t abs_tile_id, uint32_t tile_cols_per_chip, uint32_t ring_size, uint32_t ring_idx) {
    uint32_t row = abs_tile_id / (tile_cols_per_chip * ring_size);
    uint32_t id = (abs_tile_id % (tile_cols_per_chip * ring_size)) % tile_cols_per_chip;
    return tile_cols_per_chip * row + id;
}

template <bool DRAM>
inline void pack_contig_tiles_dim3_bf16(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t total = 0;
    uint32_t tile_id = 0;
    uint32_t total_cols = tile_cols_per_chip * ring_size;
    uint32_t end_abs_tile_id = dim3_rel2abs_tile_id(num_tiles - 1, tile_cols_per_chip, ring_size, my_chip_id);
    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] BEGIN pack_contig_tiles_dim3 tile_id: " << total
           << ", num_tiles: " << num_tiles << ", tile_cols_per_chip: " << tile_cols_per_chip
           << ", end_abs_tile_id: " << end_abs_tile_id << "\n";
    while (total < num_tiles) {
        uint32_t abs_tile_id = dim3_rel2abs_tile_id(tile_id, tile_cols_per_chip, ring_size, my_chip_id);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        if ((abs_tile_id >= num_banks) &&
            ((uint32_t)my_chip_id == ((abs_tile_id - num_banks) % total_cols) / tile_cols_per_chip)) {
            if ((abs_tile_id >= packet_size_in_pages * num_banks) &&
                ((uint32_t)my_chip_id ==
                 ((abs_tile_id - packet_size_in_pages * num_banks) % total_cols) / tile_cols_per_chip)) {
                cb_reserve_back(cb0_id, 2);
                // TODO: loop
                DPRINT << "\t\t\t[R][" << (uint32_t)my_chip_id << "] non contig total: " << total << ", id: " << tile_id
                       << ", abs_id: " << dim3_rel2abs_tile_id(tile_id, tile_cols_per_chip, ring_size, my_chip_id)
                       << "\n";
                noc_async_read_tile(tile_id, addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
                tile_id++;
                total++;
                noc_async_read_barrier();
                cb_push_back(cb0_id, 2);
            } else {
                DPRINT << "\t\t[R][" << (uint32_t)my_chip_id << "] SKIP!! total: " << total << ", tile_id: " << tile_id
                       << ", abs_tile_id: " << abs_tile_id << "\n";
                tile_id++;
            }
        } else {
            DPRINT << "\t\t[R][" << (uint32_t)my_chip_id << "] total: " << total << ", tile_id: " << tile_id
                   << ", abs_tile_id: " << abs_tile_id << "\n";
            cb_reserve_back(cb0_id, 2);
            if (((abs_tile_id + num_banks) <= end_abs_tile_id) &&
                (uint32_t)my_chip_id == ((abs_tile_id + num_banks) % total_cols) / tile_cols_per_chip) {
                // +12ed tile exists in same bank of output tensor
                uint32_t id = tile_id;
                for (uint32_t j = 0; j < packet_size_in_pages; j++) {
                    DPRINT << "\t\t\t[R][" << (uint32_t)my_chip_id << "] contig total: " << total << ", id: " << id
                           << ", abs_id: " << dim3_rel2abs_tile_id(id, tile_cols_per_chip, ring_size, my_chip_id)
                           << ", write_addr: " << l1_write_addr << "\n";
                    noc_async_read_tile(id, addrgen, l1_write_addr);
                    l1_write_addr += tensor0_page_size;
                    id = dim3_abs2rel_tile_id(abs_tile_id + num_banks, tile_cols_per_chip, ring_size, my_chip_id);
                }
                tile_id++;
                total += 2;
            } else {
                // TODO: loop
                DPRINT << "\t\t\t[R][" << (uint32_t)my_chip_id << "] non contig total: " << total << ", id: " << tile_id
                       << ", abs_id: " << dim3_rel2abs_tile_id(tile_id, tile_cols_per_chip, ring_size, my_chip_id)
                       << "\n";
                noc_async_read_tile(tile_id, addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
                tile_id++;
                total++;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, 2);
        }
    }
    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] DONE pack_contig_tiles_dim3 total: " << total
           << ", num_tiles: " << num_tiles << "\n";
}

template <bool DRAM>
inline void pack_llama70_t3k_prefill(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    static const uint32_t input_width = 2004;
    constexpr uint32_t contig_total =
        ((input_width / (num_banks * packet_size_in_pages)) * (num_banks * packet_size_in_pages));
    uint32_t tile_id = 0;
    uint32_t total = 0;
    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] BEGIN pack_llama70_t3k_prefill num_tiles: " << num_tiles
           << ", contig_total: " << contig_total << ", num_banks: " << num_banks
           << ", packet_size_in_pages: " << packet_size_in_pages << "\n";
    while (total < contig_total) {
        for (uint32_t i = 0; i < num_banks; i++) {
            cb_reserve_back(cb0_id, packet_size_in_pages);
            const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
            uint32_t l1_write_addr = l1_write_addr_base;
            uint32_t id = tile_id;
            for (uint32_t j = 0; j < packet_size_in_pages; j++) {
                DPRINT << "\t\t\t[R][" << (uint32_t)my_chip_id << "] contig total: " << total << ", id: " << id
                       << ", abs_id: " << dim3_rel2abs_tile_id(id, tile_cols_per_chip, ring_size, my_chip_id)
                       << ", write_addr: " << l1_write_addr << "\n";
                noc_async_read_tile(id, addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
                id += num_banks;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, packet_size_in_pages);
            tile_id++;
            total += packet_size_in_pages;
        }
        tile_id += num_banks * (packet_size_in_pages - 1);
    }

    while (total < num_tiles) {
        cb_reserve_back(cb0_id, 1);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        uint32_t id = tile_id;
        for (uint32_t j = 0; j < 1; j++) {
            DPRINT << "\t\t\t[R][" << (uint32_t)my_chip_id << "] non contig total: " << total << ", id: " << id
                   << ", abs_id: " << dim3_rel2abs_tile_id(id, tile_cols_per_chip, ring_size, my_chip_id)
                   << ", write_addr: " << l1_write_addr << "\n";
            noc_async_read_tile(id, addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
            id += num_banks;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
        tile_id++;
        total++;
    }
    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] DONE pack_llama70_t3k_prefill num_tiles: " << num_tiles
           << ", num_banks: " << num_banks << ", packet_size_in_pages: " << packet_size_in_pages << "\n";
}

template <bool DRAM>
inline void pack_full_contig_tiles(
    uint32_t num_cols, uint32_t& col_id, uint32_t tile_id_start, InterleavedAddrGenFast<DRAM>& addrgen) {
    while (col_id < num_cols) {
        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;

        DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] pack_full_contig_tiles col_id: " << col_id
               << ", end: " << num_cols << "\n";
        for (uint32_t j = 0; j < packet_size_in_pages; j++) {
            DPRINT << "\t\t[R][" << (uint32_t)my_chip_id << "] col_id: " << col_id + tile_id_start + j * num_banks
                   << ", l1_write_addr: " << l1_write_addr << ", j: " << j << "\n";
            noc_async_read_tile(col_id + tile_id_start + j * num_banks, addrgen, l1_write_addr);
            l1_write_addr += tensor0_page_size;
        }
        noc_async_read_barrier();
        col_id++;
        if (col_id % num_banks == 0) {
            col_id += num_banks * (packet_size_in_pages - 1);
        }

        cb_push_back(cb0_id, packet_size_in_pages);
    }
}

template <bool DRAM>
inline void pack_tiles_dim2_bf8(
    uint32_t filled_bank_tiles,
    uint32_t rest_full_contig_cols,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t rest_filled_bank,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    auto rest_2contig_cols = 0;
    auto rest_tiles = 0;
    if (num_banks * 3 < rest_filled_bank) {
        rest_2contig_cols = (num_banks - rest_full_contig_cols);
        rest_tiles = rest_2contig_cols;
    } else if (num_banks * 2 <= rest_filled_bank) {
        rest_2contig_cols = num_banks;
        rest_tiles = (rest_filled_bank) % (num_banks * 2);
    } else if (num_banks < rest_filled_bank) {
        rest_2contig_cols = (rest_filled_bank) % num_banks;
        rest_tiles = num_banks - rest_2contig_cols;
    } else {
        rest_tiles = rest_filled_bank;
    }
    DPRINT << "[R][" << (uint32_t)my_chip_id << "] pack_tiles_dim2_bf8 filled_bank_tiles:" << filled_bank_tiles
           << ", rest_2contig_cols:" << rest_2contig_cols << ", rest_tiles:" << rest_tiles << "\n";
    uint32_t outer_id = 0;
    while (total < rest_2contig_cols + (filled_bank_tiles + rest_full_contig_cols)) {
        uint32_t num_2contig = min(rest_2contig_cols - outer_id, 2);
        DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] total: " << total
               << ", end: " << rest_2contig_cols + (filled_bank_tiles + rest_full_contig_cols)
               << ", num_2contig: " << num_2contig << "\n";

        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;

        uint32_t id = total + tile_id_start;
        for (uint32_t k = 0; k < num_2contig; k++) {
            for (uint32_t j = 0; j < 2; j++) {
                DPRINT << "\t\t[R][" << (uint32_t)my_chip_id << "] tile_id: " << id + j * num_banks
                       << ", l1_write_addr: " << l1_write_addr << ", j: " << j << "\n";
                noc_async_read_tile(id + j * num_banks, tensor0_addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
            }
            id++;
        }
        outer_id += num_2contig;
        total += num_2contig;
        if (total % num_banks == 0) {
            total += num_banks + rest_full_contig_cols;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
    }
    total += tile_id_start;
    pack_non_contig_tiles<DRAM>(rest_tiles, total, tensor0_addrgen);
}

template <bool DRAM>
inline void pack_tiles_dim2_bf16(
    uint32_t num_tiles_per_chip,
    uint32_t filled_bank_rows,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t rest_filled_bank,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    auto rest_tiles = 0;
    if (num_banks * 1 < rest_filled_bank) {
        rest_tiles = num_banks - (rest_filled_bank % (num_banks * (packet_size_in_pages - 1)));
    } else {
        rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    }
    DPRINT << "[R][" << (uint32_t)my_chip_id << "] filled_bank_rows:" << filled_bank_rows
           << ", rest_tiles:" << rest_tiles << "\n";
    total += tile_id_start;
    pack_non_contig_tiles<DRAM>(rest_tiles, total, tensor0_addrgen);
}

template <bool DRAM>
inline void pack_tiles_dim2(
    uint32_t num_tiles_per_chip, uint32_t tile_id_start, InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    uint32_t total = 0;
    auto filled_bank_rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
    auto rest_filled_bank = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    auto filled_bank_tiles = filled_bank_rows * num_banks * packet_size_in_pages;
    auto rest_full_contig_cols = 0;
    if (num_banks * (packet_size_in_pages - 1) < rest_filled_bank) {
        rest_full_contig_cols = rest_filled_bank % (num_banks * (packet_size_in_pages - 1));
    }
    DPRINT << "[R][" << (uint32_t)my_chip_id << "] pack_tiles_dim2 filled_bank_rows:" << filled_bank_rows
           << ", rest_filled_bank: " << rest_filled_bank << ", rest_full_contig_cols: " << rest_full_contig_cols
           << "\n";

    pack_full_contig_tiles<DRAM>(filled_bank_tiles + rest_full_contig_cols, total, tile_id_start, tensor0_addrgen);

    if constexpr (packet_size_in_pages == 2) {
        pack_tiles_dim2_bf16<DRAM>(
            num_tiles_per_chip, filled_bank_rows, tile_id_start, total, rest_filled_bank, tensor0_addrgen);
    } else {
        pack_tiles_dim2_bf8<DRAM>(
            filled_bank_tiles, rest_full_contig_cols, tile_id_start, total, rest_filled_bank, tensor0_addrgen);
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
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_cols_per_chip = get_arg_val<uint32_t>(arg_idx++);

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "last_dim: " << (uint32_t)last_dim << "\n";
    DPRINT << "num_banks: " << (uint32_t)num_banks << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "[R][" << (uint32_t)my_chip_id << "] tile_id_start: " << (uint32_t)tile_id_start << "\n";
    DPRINT << "tile_id_end: " << (uint32_t)tile_id_end << "\n";
    DPRINT << "num_tiles_per_chip: " << (uint32_t)num_tiles_per_chip << "\n";
    DPRINT << "ring_size: " << (uint32_t)ring_size << "\n";
    DPRINT << "tile_cols_per_chip: " << (uint32_t)tile_cols_per_chip << "\n";

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet size in pages: " << (uint32_t)packet_size_in_pages << "\n";

    if constexpr (last_dim) {
        if constexpr (true) {
            if constexpr (packet_size_in_pages == 2) {
                pack_contig_tiles_dim3_bf16<is_dram>(
                    num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            } else {
                pack_llama70_t3k_prefill<is_dram>(num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            }
        } else {
            pack_non_contig_tiles<is_dram>(tile_id_end - tile_id_start, tile_id_start, tensor0_addrgen);
        }
    } else {
        pack_tiles_dim2(num_tiles_per_chip, tile_id_start, tensor0_addrgen);
    }

    DPRINT << "[R] DONE \n";
}

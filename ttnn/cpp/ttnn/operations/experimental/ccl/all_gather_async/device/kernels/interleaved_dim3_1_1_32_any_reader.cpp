// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "minimal_ccl_common.hpp"
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
constexpr uint32_t bf8_dim3_type = get_compile_time_arg_val(7);

template <bool DRAM>
inline void pack_contig_tiles_dim3_bf16(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t total = 0;
    uint32_t tile_id = 0;
    uint32_t total_cols = tile_cols_per_chip * ring_size;
    uint32_t end_abs_tile_id = dim3_rel2abs_tile_id(num_tiles - 1, tile_cols_per_chip, ring_size, my_chip_id);
    while (total < num_tiles) {
        uint32_t abs_tile_id = dim3_rel2abs_tile_id(tile_id, tile_cols_per_chip, ring_size, my_chip_id);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;
        if (dim3_was_stride_sent(abs_tile_id, total_cols, tile_cols_per_chip, num_banks, my_chip_id)) {
            if (dim3_was_stride_sent(
                    abs_tile_id, total_cols, tile_cols_per_chip, packet_size_in_pages * num_banks, my_chip_id)) {
                cb_reserve_back(cb0_id, packet_size_in_pages);
                noc_async_read_tile(tile_id, addrgen, l1_write_addr);
                tile_id++;
                total++;
                noc_async_read_barrier();
                cb_push_back(cb0_id, packet_size_in_pages);
            } else {
                tile_id++;  // skip tile as it is already processed
            }
        } else {
            cb_reserve_back(cb0_id, packet_size_in_pages);
            // check whether there is contiguous tile, the tile is in the local chip/buffer
            if ((abs_tile_id + num_banks) <= end_abs_tile_id &&
                dim3_is_tile_in_local(abs_tile_id + num_banks, total_cols, tile_cols_per_chip, my_chip_id)) {
                // +12ed tile exists in same bank of output tensor
                uint32_t id = tile_id;
                for (uint32_t j = 0; j < packet_size_in_pages; j++) {
                    noc_async_read_tile(id, addrgen, l1_write_addr);
                    l1_write_addr += tensor0_page_size;
                    id = dim3_abs2rel_tile_id(abs_tile_id + num_banks, tile_cols_per_chip, ring_size, my_chip_id);
                }
                tile_id++;
                total += packet_size_in_pages;
            } else {
                // TODO: loop
                noc_async_read_tile(tile_id, addrgen, l1_write_addr);
                tile_id++;
                total++;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, packet_size_in_pages);
        }
    }
}

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
inline void pack_llama_8b_n300(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    static const uint32_t input_width = 2004;
    constexpr uint32_t num_full_contig = 41 * 12;
    constexpr uint32_t num_2contig = 12;
    constexpr uint32_t rest_tiles = 12;
    uint32_t tile_id = 0;
    uint32_t total = 0;
    pack_full_contig(num_full_contig, tile_id, addrgen);
    pack_2contig_bf8(num_2contig, tile_id, addrgen);
    pack_non_contig(rest_tiles, tile_id, addrgen);
}

template <bool DRAM>
inline void pack_dim3_bf16_rest16_optimized(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t tile_id = 0;
    const uint32_t num_contig2 = 4;
    uint32_t row = num_tiles / tile_cols_per_chip;
    // 32768 -> 128 tiles. 24x5 + 8
    for (uint32_t i = 0; i < row; i++) {
        pack_full_contig(num_contig2, tile_id, addrgen);
        pack_non_contig(8, tile_id, addrgen);
        tile_id += 4;
    }
}

template <bool DRAM>
inline void pack_dim3_bf16_rest8_optimized(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t tile_id = 0;
    const uint32_t num_contig2 = num_banks * 5;
    uint32_t row = num_tiles / tile_cols_per_chip;
    // 32768 -> 128 tiles. 24x5 + 8
    if constexpr (my_chip_id % 3 == 0) {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                pack_full_contig(num_contig2, tile_id, addrgen);
                pack_non_contig(8, tile_id, addrgen);
            } else if (i % 3 == 1) {
                pack_non_contig(8, tile_id, addrgen);
                pack_full_contig(num_contig2, tile_id, addrgen);
            } else {
                pack_non_contig(4, tile_id, addrgen);
                pack_full_contig(num_contig2, tile_id, addrgen);
                pack_non_contig(4, tile_id, addrgen);
            }
        }
    } else if constexpr (my_chip_id % 3 == 1) {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                pack_non_contig(4, tile_id, addrgen);
                pack_full_contig(num_contig2, tile_id, addrgen);
                pack_non_contig(4, tile_id, addrgen);
            } else if (i % 3 == 1) {
                pack_full_contig(num_contig2, tile_id, addrgen);
                pack_non_contig(8, tile_id, addrgen);
            } else {
                pack_non_contig(8, tile_id, addrgen);
                pack_full_contig(num_contig2, tile_id, addrgen);
            }
        }
    } else {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                pack_non_contig(8, tile_id, addrgen);
                pack_full_contig(num_contig2, tile_id, addrgen);
            } else if (i % 3 == 1) {
                pack_non_contig(4, tile_id, addrgen);
                pack_full_contig(num_contig2, tile_id, addrgen);
                pack_non_contig(4, tile_id, addrgen);
            } else {
                pack_full_contig(num_contig2, tile_id, addrgen);
                pack_non_contig(8, tile_id, addrgen);
            }
        }
    }
}

template <bool DRAM>
inline void pack_dim3_bf8_reminder32(
    uint32_t num_tiles, uint32_t ring_size, uint32_t tile_cols_per_chip, InterleavedAddrGenFast<DRAM>& addrgen) {
    uint32_t tile_id = 0;
    uint32_t row = num_tiles / tile_cols_per_chip;
    uint32_t num_full_contig = (tile_cols_per_chip / (num_banks * packet_size_in_pages)) * num_banks;
    uint32_t num_contig2 =
        ((tile_cols_per_chip - num_full_contig * packet_size_in_pages) / (num_banks * 2)) * num_banks;
    if constexpr (my_chip_id % 3 == 0) {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
                pack_non_contig(8, tile_id, addrgen);
            } else if (i % 3 == 1) {
                pack_non_contig(8, tile_id, addrgen);
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
            } else {
                pack_non_contig(4, tile_id, addrgen);
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
                pack_non_contig(4, tile_id, addrgen);
            }
        }
    } else if constexpr (my_chip_id % 3 == 1) {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                pack_non_contig(4, tile_id, addrgen);
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
                pack_non_contig(4, tile_id, addrgen);
            } else if (i % 3 == 1) {
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
                pack_non_contig(8, tile_id, addrgen);
            } else {
                pack_non_contig(8, tile_id, addrgen);
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
            }
        }
    } else {
        for (uint32_t i = 0; i < row; i++) {
            if (i % 3 == 0) {
                pack_non_contig(8, tile_id, addrgen);
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
            } else if (i % 3 == 1) {
                pack_non_contig(4, tile_id, addrgen);
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
                pack_non_contig(4, tile_id, addrgen);
            } else {
                pack_full_contig(num_full_contig, tile_id, addrgen);
                pack_2contig_bf8(num_contig2, tile_id, addrgen);
                pack_non_contig(8, tile_id, addrgen);
            }
        }
    }
}

template <bool DRAM>
inline void pack_tiles_dim2_bf8(
    uint32_t filled_bank_tiles,
    uint32_t rest_full_contig_ids,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t rest_tiles,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    uint32_t rest_half_contig_ids, rest_orphan_tiles;
    if (num_banks * 3 < rest_tiles) {
        rest_half_contig_ids = (num_banks - rest_full_contig_ids);
        rest_orphan_tiles = rest_half_contig_ids;
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
    uint32_t num_tiles = rest_half_contig_ids + (filled_bank_tiles + rest_full_contig_ids);
    uint32_t outer_id = 0;

    while (total < num_tiles) {
        uint32_t num_2contig = min(rest_half_contig_ids - outer_id, 2);

        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;

        uint32_t id = total + tile_id_start;
        for (uint32_t k = 0; k < num_2contig; k++) {
            for (uint32_t j = 0; j < 2; j++) {
                noc_async_read_tile(id + j * num_banks, tensor0_addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
            }
            id++;
        }
        outer_id += num_2contig;
        total += num_2contig;
        if (total % num_banks == 0) {
            total += num_banks + rest_full_contig_ids;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
    }
    total += tile_id_start;
    pack_non_contig(rest_orphan_tiles, total, tensor0_addrgen);
}

template <bool DRAM>
inline void pack_tiles_dim2_bf16(
    uint32_t num_tiles_per_chip,
    uint32_t filled_bank_rows,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t rest_tiles,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    auto rest_orphan_tiles = 0;
    if (num_banks * 1 < rest_tiles) {
        rest_orphan_tiles = num_banks - (rest_tiles % (num_banks * (packet_size_in_pages - 1)));
    } else {
        rest_orphan_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    }
    total += tile_id_start;
    pack_non_contig(rest_orphan_tiles, total, tensor0_addrgen);
}

template <bool DRAM>
inline void pack_tiles_dim2(
    uint32_t num_tiles_per_chip, uint32_t tile_id_start, InterleavedAddrGenFast<DRAM>& tensor0_addrgen) {
    uint32_t total = 0;
    auto filled_bank_rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
    auto rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    auto filled_bank_tiles = filled_bank_rows * num_banks * packet_size_in_pages;
    auto rest_full_contig_ids = 0;
    if (num_banks * (packet_size_in_pages - 1) < rest_tiles) {
        rest_full_contig_ids = rest_tiles % (num_banks * (packet_size_in_pages - 1));
    }
    pack_full_contig(filled_bank_rows * num_banks + rest_full_contig_ids, total, tensor0_addrgen);
    if constexpr (packet_size_in_pages == 2) {
        pack_tiles_dim2_bf16<DRAM>(
            num_tiles_per_chip, filled_bank_rows, tile_id_start, total, rest_tiles, tensor0_addrgen);
    } else {
        pack_tiles_dim2_bf8<DRAM>(
            filled_bank_tiles, rest_full_contig_ids, tile_id_start, total, rest_tiles, tensor0_addrgen);
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
    DPRINT << "bf8_dim3_type: " << (uint32_t)bf8_dim3_type << "\n";

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
        if constexpr (packet_size_in_pages == 2) {
            if (is_dram && tile_cols_per_chip == 128) {
                pack_dim3_bf16_rest8_optimized(num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            } else if (is_dram && tile_cols_per_chip == 16) {
                pack_dim3_bf16_rest16_optimized(num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            } else {
                pack_contig_tiles_dim3_bf16<is_dram>(
                    num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            }
        } else {
            if constexpr ((BF8_DIM3_TYPE)bf8_dim3_type == BF8_DIM3_REMAINDER_32) {
                pack_dim3_bf8_reminder32<is_dram>(num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            } else if constexpr ((BF8_DIM3_TYPE)bf8_dim3_type == LLAMA_8B_N300) {
                pack_llama_8b_n300<is_dram>(num_tiles_per_chip, ring_size, tile_cols_per_chip, tensor0_addrgen);
            } else {
                // assert or handle default case
            }
        }
    } else {
        pack_tiles_dim2(num_tiles_per_chip, tile_id_start, tensor0_addrgen);
    }

    DPRINT << "[R] DONE \n";
}

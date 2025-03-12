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
constexpr uint32_t dim = get_compile_time_arg_val(5);
constexpr uint32_t num_banks = get_compile_time_arg_val(6);

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

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "dim: " << (uint32_t)dim << "\n";
    DPRINT << "num_banks: " << (uint32_t)num_banks << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "[R][" << (uint32_t)my_chip_id << "] tile_id_start: " << (uint32_t)tile_id_start << "\n";
    DPRINT << "tile_id_end: " << (uint32_t)tile_id_end << "\n";
    DPRINT << "num_tiles_per_chip: " << (uint32_t)num_tiles_per_chip << "\n";

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet size in pages: " << (uint32_t)packet_size_in_pages << "\n";
    if constexpr (true) {
        uint32_t total = 0;
        if constexpr (dim == 1 || dim == 2) {
            // data layout on 12 banks
            // bank   0  1 ..  10 11
            //      | 0| 1| .. 10|11|
            // tile |12|13| .. 22|23|
            //      |24|25| .. 34|35|
            //      |36|37|38| (empt)
            // -> filled_bank_2rows = 1
            // -> filled_bank_1rows = 3
            // -> rest_contig_tiles = 3
            // -> rest_tiles        = 9

            if (packet_size_in_pages == 2) {  // bf16
                auto filled_bank_2rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
                auto filled_bank_1rows = num_tiles_per_chip / num_banks;
                auto rest_contig_tiles =
                    filled_bank_1rows % packet_size_in_pages != 0 ? num_tiles_per_chip % num_banks : 0;
                auto rest_tiles = filled_bank_1rows % packet_size_in_pages != 0 ? num_banks - rest_contig_tiles
                                                                                : num_tiles_per_chip % num_banks;
                DPRINT << "[R][" << (uint32_t)my_chip_id << "] filled_bank_2rows:" << filled_bank_2rows
                       << ", filled_bank_1rows:" << filled_bank_1rows << ", rest_contig_tiles:" << rest_contig_tiles
                       << ", rest_tiles:" << rest_tiles << "\n";

                while (total < filled_bank_2rows * num_banks * packet_size_in_pages + rest_contig_tiles) {
                    cb_reserve_back(cb0_id, packet_size_in_pages);
                    const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
                    uint32_t l1_write_addr = l1_write_addr_base;

                    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] total: " << total
                           << ", end: " << filled_bank_2rows * num_banks * packet_size_in_pages + rest_contig_tiles
                           << "\n";
                    for (uint32_t j = 0; j < packet_size_in_pages; j++) {
                        DPRINT << "\t\t[R][" << (uint32_t)my_chip_id
                               << "] tile_id: " << total + tile_id_start + j * num_banks
                               << ", l1_write_addr: " << l1_write_addr << ", j: " << j << "\n";
                        noc_async_read_tile(total + tile_id_start + j * num_banks, tensor0_addrgen, l1_write_addr);
                        l1_write_addr += tensor0_page_size;
                    }
                    noc_async_read_barrier();
                    total++;
                    if (total % num_banks == 0) {
                        total += num_banks * (packet_size_in_pages - 1);  // * 1
                    }

                    cb_push_back(cb0_id, packet_size_in_pages);
                }
                for (uint32_t i = 0; i < rest_tiles; i += packet_size_in_pages) {
                    uint32_t num_pages_to_read = min(rest_tiles - i, packet_size_in_pages);
                    cb_reserve_back(cb0_id, num_pages_to_read);
                    const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
                    uint32_t l1_write_addr = l1_write_addr_base;
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        DPRINT << "[R][" << (uint32_t)my_chip_id
                               << "] rest_tiles tile_id: " << total + tile_id_start + j
                               << ", tile_id_start:" << tile_id_start << "\n";
                        noc_async_read_tile(total + tile_id_start + j, tensor0_addrgen, l1_write_addr);
                        l1_write_addr += tensor0_page_size;
                    }
                    noc_async_read_barrier();
                    total += num_pages_to_read;
                    cb_push_back(cb0_id, num_pages_to_read);
                }
            } else {  // bf8
                auto filled_bank_4rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
                auto filled_bank_2rows =
                    (num_tiles_per_chip % (num_banks * packet_size_in_pages)) / (num_banks * packet_size_in_pages / 2);
                auto rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages / 2);  // rest of 2 bank row
                DPRINT << "[R][" << (uint32_t)my_chip_id << "] filled_bank_4rows:" << filled_bank_4rows
                       << ", filled_bank_2rows:" << filled_bank_2rows << ", rest_tiles:" << rest_tiles << "\n";

                while (total < filled_bank_4rows * num_banks * packet_size_in_pages) {
                    cb_reserve_back(cb0_id, packet_size_in_pages);
                    const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
                    uint32_t l1_write_addr = l1_write_addr_base;

                    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] total: " << total
                           << ", end: " << filled_bank_4rows * num_banks * packet_size_in_pages << "\n";
                    for (uint32_t j = 0; j < packet_size_in_pages; j++) {
                        DPRINT << "\t\t[R][" << (uint32_t)my_chip_id
                               << "] tile_id: " << total + tile_id_start + j * num_banks
                               << ", l1_write_addr: " << l1_write_addr << ", j: " << j << "\n";
                        noc_async_read_tile(total + tile_id_start + j * num_banks, tensor0_addrgen, l1_write_addr);
                        l1_write_addr += tensor0_page_size;
                    }
                    noc_async_read_barrier();
                    total++;
                    if (total % num_banks == 0) {
                        total += num_banks * (packet_size_in_pages - 1);  // * 3
                    }
                    cb_push_back(cb0_id, packet_size_in_pages);
                }
                while (total < filled_bank_2rows * num_banks * (packet_size_in_pages / 2) +
                                   (filled_bank_4rows * num_banks * packet_size_in_pages)) {
                    cb_reserve_back(cb0_id, packet_size_in_pages);
                    const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
                    uint32_t l1_write_addr = l1_write_addr_base;

                    DPRINT << "\t[R][" << (uint32_t)my_chip_id << "] total: " << total << ", end: "
                           << filled_bank_2rows * num_banks * (packet_size_in_pages / 2) +
                                  (filled_bank_4rows * num_banks * packet_size_in_pages)
                           << "\n";
                    for (uint32_t j = 0; j < packet_size_in_pages / 2; j++) {
                        DPRINT << "\t\t[R][" << (uint32_t)my_chip_id
                               << "] tile_id: " << total + tile_id_start + j * num_banks
                               << ", l1_write_addr: " << l1_write_addr << ", j: " << j << "\n";
                        noc_async_read_tile(total + tile_id_start + j * num_banks, tensor0_addrgen, l1_write_addr);
                        l1_write_addr += tensor0_page_size;
                    }
                    total++;
                    for (uint32_t j = 0; j < packet_size_in_pages / 2; j++) {
                        DPRINT << "\t\t[R][" << (uint32_t)my_chip_id
                               << "] tile_id: " << total + tile_id_start + j * num_banks
                               << ", l1_write_addr: " << l1_write_addr << ", j: " << j << "\n";
                        noc_async_read_tile(total + tile_id_start + j * num_banks, tensor0_addrgen, l1_write_addr);
                        l1_write_addr += tensor0_page_size;
                    }

                    noc_async_read_barrier();
                    total++;
                    if (total % num_banks == 0) {
                        total += num_banks * (packet_size_in_pages - 3);  // * 1
                    }
                    cb_push_back(cb0_id, packet_size_in_pages);
                }
                for (uint32_t i = 0; i < rest_tiles; i += packet_size_in_pages) {
                    uint32_t num_pages_to_read =
                        min(rest_tiles - i, packet_size_in_pages);  // rest_tiles % packet_size_in_pages is faster?
                    cb_reserve_back(cb0_id, num_pages_to_read);
                    const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
                    uint32_t l1_write_addr = l1_write_addr_base;
                    for (uint32_t j = 0; j < num_pages_to_read; j++) {
                        DPRINT << "\t\t[R][" << (uint32_t)my_chip_id
                               << "] rest_tiles tile_id: " << total + tile_id_start + j
                               << ", tile_id_start:" << tile_id_start << "\n";
                        noc_async_read_tile(total + tile_id_start + j, tensor0_addrgen, l1_write_addr);
                        l1_write_addr += tensor0_page_size;
                    }
                    noc_async_read_barrier();
                    total += num_pages_to_read;
                    cb_push_back(cb0_id, num_pages_to_read);
                }
            }
        } else {
            DPRINT << "NOT IMPLEMENTED YET!!!\n";
        }
    } else {
        uint32_t tile_id = tile_id_start;
        while (tile_id < tile_id_end) {
            DPRINT << "tile_id: " << tile_id << "\n";
            cb_reserve_back(cb0_id, packet_size_in_pages);
            const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
            uint32_t l1_write_addr = l1_write_addr_base;

            uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
            for (uint32_t j = 0; j < num_pages_to_read; j++) {
                noc_async_read_tile(tile_id, tensor0_addrgen, l1_write_addr);
                l1_write_addr += tensor0_page_size;
                tile_id++;
            }

            noc_async_read_barrier();
            cb_push_back(cb0_id, packet_size_in_pages);
        }
    }

    DPRINT << "[R] DONE \n";
}

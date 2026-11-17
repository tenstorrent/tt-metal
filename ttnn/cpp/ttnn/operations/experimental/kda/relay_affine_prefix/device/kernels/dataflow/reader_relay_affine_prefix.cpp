// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/chronology.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/dataflow/packed_summary.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/kda/device/kernels/persistent_counter.hpp"

constexpr uint32_t cb_e = get_compile_time_arg_val(0);
constexpr uint32_t cb_b = get_compile_time_arg_val(1);
constexpr uint32_t cb_ein = get_compile_time_arg_val(2);
constexpr uint32_t cb_chron = get_compile_time_arg_val(3);
constexpr uint32_t sp_size = get_compile_time_arg_val(4);
constexpr uint32_t local_rows = get_compile_time_arg_val(5);
constexpr uint32_t kt = get_compile_time_arg_val(6);  // key tiles; the state is kt x kt tiles
// cb_ein's storage again, as compute's exact (UnpackToDest) view of the entry state.
constexpr uint32_t cb_ein_exact = get_compile_time_arg_val(7);
// One BF16 32x32 identity tile, for compute's exact I @ B.
constexpr uint32_t cb_identity = get_compile_time_arg_val(8);
// Slab (slab_kernel_args): the initial state is entry `index * slab_scale + slab_offset`; zeros at actual_start == 0.
constexpr bool slab = get_compile_time_arg_val(9);
constexpr uint32_t slab_index_limit = get_compile_time_arg_val(10);
constexpr uint32_t slab_scale = get_compile_time_arg_val(11);
constexpr uint32_t slab_offset = get_compile_time_arg_val(12);
constexpr uint32_t entry_tiles = get_compile_time_arg_val(13);
// cb_chron's second DRAM-aligned slot, where the slab index lands; actual_start is word 0.
constexpr uint32_t slab_index_word = get_compile_time_arg_val(14);
constexpr auto packed_args = TensorAccessorArgs<15>();
constexpr auto init_args = TensorAccessorArgs<packed_args.next_compile_time_args_offset()>();
constexpr auto start_args = TensorAccessorArgs<init_args.next_compile_time_args_offset()>();
constexpr auto slab_index_args = TensorAccessorArgs<start_args.next_compile_time_args_offset()>();
constexpr uint32_t state_tiles = kt * kt;

// Head core: this rank's complement-form [E | B] rows of the head (E = A - I), then its entry state column by column
// (the initial state on the chronologically first rank, else the predecessor's landed columns) in both of its views.
void kernel_main() {
    size_t arg = 0;
    const uint32_t packed_address = get_common_arg_val<uint32_t>(0);
    const uint32_t init_address = get_common_arg_val<uint32_t>(1);
    const uint32_t start_address = get_common_arg_val<uint32_t>(2);
    const uint32_t slab_index_address = get_common_arg_val<uint32_t>(3);
    const uint32_t head = get_arg_val<uint32_t>(arg++);
    const uint32_t arrival_semaphore = get_arg_val<uint32_t>(arg++);
    const uint32_t sp_rank = get_arg_val<uint32_t>(arg++);

    cb_reserve_back(cb_chron, 1);
    const auto start = TensorAccessor(start_args, start_address);
    noc_async_read(start.get_noc_addr(0), get_write_ptr(cb_chron), sizeof(uint32_t));
    if constexpr (slab) {
        const auto slab_index = TensorAccessor(slab_index_args, slab_index_address);
        noc_async_read(
            slab_index.get_noc_addr(0), get_write_ptr(cb_chron) + slab_index_word * sizeof(uint32_t), sizeof(uint32_t));
    }
    const auto packed = TensorAccessor(packed_args, packed_address);
    cb_reserve_back(cb_e, state_tiles);
    cb_reserve_back(cb_b, state_tiles);
    const uint32_t eb_tile = get_tile_size(cb_e);
    // [E | B] rows of this head: kt tile rows of 2 kt tiles.
    for (uint32_t r = 0; r < kt; ++r) {
        for (uint32_t c = 0; c < kt; ++c) {
            const uint32_t page = (head * kt + r) * 2 * kt + c;
            noc_async_read_page(page, packed, get_write_ptr(cb_e) + (r * kt + c) * eb_tile);
            noc_async_read_page(page + kt, packed, get_write_ptr(cb_b) + (r * kt + c) * eb_tile);
        }
    }
    // The BF16 identity tile while the reads are in flight.
    cb_reserve_back(cb_identity, 1);
    auto* identity = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(cb_identity));
    for (uint32_t i = 0; i < tt::constants::TILE_HW; ++i) {
        identity[i] = 0;
    }
    kda::write_identity_diagonal(identity);
    cb_push_back(cb_identity, 1);
    noc_async_read_barrier();
    const auto* chron_words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_chron));
    const uint32_t actual_start = chron_words[0];
    const uint32_t index = slab ? chron_words[slab_index_word] : 0;
    const auto topology = kda_chronology::derive(actual_start, sp_rank, sp_size, local_rows);
    const uint32_t position = (topology.rank + sp_size - topology.first_rank) % sp_size;
    cb_push_back(cb_chron, 1);
    cb_push_back(cb_e, state_tiles);
    cb_push_back(cb_b, state_tiles);

    // The entry state is stored column-major by tile, so each tile column (all the next state's column needs) can
    // land, be computed and leave on its own.
    cb_reserve_back(cb_ein, state_tiles);
    cb_reserve_back(cb_ein_exact, state_tiles);
    if (position == 0) {
        const uint32_t state_tile = get_tile_size(cb_ein);
        if (slab && (actual_start == 0 || index >= slab_index_limit)) {
            const uint64_t zeros = get_noc_addr(MEM_ZEROS_BASE);
            for (uint32_t offset = 0; offset < state_tiles * state_tile; offset += MEM_ZEROS_SIZE) {
                noc_async_read(zeros, get_write_ptr(cb_ein) + offset, MEM_ZEROS_SIZE);
            }
        } else {
            const auto init = TensorAccessor(init_args, init_address);
            const uint32_t entry_base = (index * slab_scale + slab_offset) * entry_tiles;
            for (uint32_t j = 0; j < kt; ++j) {
                for (uint32_t r = 0; r < kt; ++r) {
                    noc_async_read_page(
                        entry_base + head * state_tiles + r * kt + j,
                        init,
                        get_write_ptr(cb_ein) + (j * kt + r) * state_tile);
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_ein, state_tiles);
        cb_push_back(cb_ein_exact, state_tiles);
    } else {
        // The predecessor's sender writes each tile column of this head's entry state into cb_ein's storage, then
        // counts it.
        for (uint32_t j = 0; j < kt; ++j) {
            kda_counter::wait(arrival_semaphore, j + 1);
            cb_push_back(cb_ein, kt);
            cb_push_back(cb_ein_exact, kt);
        }
        kda_counter::consume(arrival_semaphore, kt);
    }
}

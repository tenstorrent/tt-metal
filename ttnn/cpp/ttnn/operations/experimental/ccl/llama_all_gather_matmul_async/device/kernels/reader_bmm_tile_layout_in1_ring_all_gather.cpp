// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"
#include "api/remote_circular_buffer.h"
#include "api/debug/dprint.h"
#include "api/debug/dprint_tile.h"
#include "api/tensor/noc_traits.h"

enum class CORE_TYPE : uint8_t { IDLE_CORE = 0, WORKER_CORE = 1, HOP_CORE = 2 };

template <typename TensorAccessorType>
void read_block_from_dram(
    Noc& noc,
    DataflowBuffer& cb,
    const TensorAccessorType& s1,
    uint32_t tensor_width_in_tiles,
    uint32_t block_w_idx,
    uint32_t block_h_idx,
    uint32_t block_w_t,
    uint32_t block_h_t,
    uint32_t tile_size_bytes) {
    uint32_t l1_write_addr = cb.get_write_ptr();

    // Horizontal idx + vertical idx * width = row major index
    uint32_t block_tile_id = block_w_idx * block_w_t + (block_h_idx * block_h_t) * tensor_width_in_tiles;
    for (uint32_t h = 0; h < block_h_t; ++h) {
        uint32_t tile_id = block_tile_id + h * tensor_width_in_tiles;
        for (uint32_t w = 0; w < block_w_t; ++w) {
            noc.async_read(s1, CoreLocalMem<uint8_t>(l1_write_addr), tile_size_bytes, {.page_id = tile_id + w}, {});
            l1_write_addr += tile_size_bytes;
        }
    }
    noc.async_read_barrier();
}

void do_signaling(Noc& noc, uint32_t& rt_args_idx) {
    const uint32_t pv_core_x = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t pv_core_y = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t pv_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    Semaphore<> pv_sem(pv_semaphore_id);
    const bool is_privilaged = get_arg_val<uint32_t>(rt_args_idx++) == 1;
    if (is_privilaged) {
        const uint32_t target_sem_value = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t multicast_start_x = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t multicast_start_y = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t multicast_end_x = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t multicast_end_y = get_arg_val<uint32_t>(rt_args_idx++);
        const uint32_t num_signalling_semaphores = get_arg_val<uint32_t>(rt_args_idx++);
        Semaphore<> rs_sem(get_arg_val<uint32_t>(rt_args_idx++));
        pv_sem.wait(target_sem_value);
        pv_sem.set(1);
        // Relay pv_sem's value into the (different) rs_sem over the reduce-scatter core range.
        // noc is the reader's NoC (NOC_0)
        pv_sem.relay_multicast(
            noc,
            rs_sem,
            multicast_start_x,
            multicast_start_y,
            multicast_end_x,
            multicast_end_y,
            num_signalling_semaphores);
    } else {
        pv_sem.up(noc, pv_core_x, pv_core_y, 1);
    }
}

void kernel_main() {
    // Compile time args
    constexpr const bool in1_is_dram_interleaved = get_compile_time_arg_val(0);
    constexpr const bool in1_is_dram_sharded = get_compile_time_arg_val(1);
    constexpr uint32_t in1_block_height_in_tiles = get_compile_time_arg_val(2);  // Padded block shape
    constexpr uint32_t in1_block_width_in_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_tensor_width_in_tiles = get_compile_time_arg_val(4);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(5);
    constexpr uint32_t batch = get_compile_time_arg_val(6);
    constexpr uint32_t in1_block_page_size = get_compile_time_arg_val(7);
    constexpr uint32_t in1_block_page_size_last = get_compile_time_arg_val(8);
    constexpr uint32_t in1_block_width_num_pages = get_compile_time_arg_val(9);
    constexpr uint32_t in1_shard_width_in_dram = get_compile_time_arg_val(10);

    Noc noc_obj;

    uint32_t rt_args_idx = 0;
    constexpr bool needs_signaler = get_compile_time_arg_val(15) == 1;
    uint32_t core_type = get_arg_val<uint32_t>(rt_args_idx++);
    if (core_type == (uint32_t)CORE_TYPE::IDLE_CORE || core_type == (uint32_t)CORE_TYPE::HOP_CORE) {
        if constexpr (needs_signaler) {
            do_signaling(noc_obj, rt_args_idx);
            noc_obj.async_write_barrier();
        }
        return;
    }
    const uint32_t in1_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    const uint32_t ring_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dram_bank_id = 0;
    uint32_t vc = 0;
    uint32_t dram_read_offset = 0;

    if constexpr (in1_is_dram_interleaved || in1_is_dram_sharded) {
        dram_bank_id = get_arg_val<uint32_t>(rt_args_idx++);
        vc = get_arg_val<uint32_t>(rt_args_idx++);
        dram_read_offset = get_arg_val<uint32_t>(rt_args_idx++);
    }

    constexpr uint32_t cb_id_in1 = get_compile_time_arg_val(11);
    constexpr uint32_t sync_cb = get_compile_time_arg_val(12);
    constexpr uint32_t sync_cb2 = get_compile_time_arg_val(13);
    constexpr uint32_t remote_cb_id = get_compile_time_arg_val(14);

    DataflowBuffer cb_in1(cb_id_in1);
    DataflowBuffer cb_sync(sync_cb);
    DataflowBuffer cb_sync2(sync_cb2);

    constexpr auto src_args = TensorAccessorArgs<16>();

    const uint32_t in1_block_num_tiles = in1_block_height_in_tiles * in1_block_width_in_tiles;

    // Address setup
    constexpr const uint32_t in1_tile_hw = get_tile_hw(cb_id_in1);
    constexpr uint32_t in1_single_tile_size_bytes = get_tile_size(cb_id_in1);
    constexpr DataFormat in1_data_format = get_dataformat(cb_id_in1);
    const auto s1 = TensorAccessor(src_args, in1_tensor_addr);

    uint32_t in1_shard_width_offset_bytes = 0;
    uint32_t in1_dram_shard_block_size_bytes = 0;
    uint32_t dram_read_offset_bytes = 0;
    uint32_t l1_write_addr_in1;
    uint32_t l1_read_addr_in1 = 0;

    if constexpr (in1_is_dram_sharded) {
        in1_shard_width_offset_bytes = in1_shard_width_in_dram * in1_single_tile_size_bytes;
        in1_dram_shard_block_size_bytes = in1_shard_width_offset_bytes * in1_block_height_in_tiles;
        dram_read_offset_bytes = dram_read_offset * in1_block_width_in_tiles * in1_single_tile_size_bytes;
    }

    for (uint32_t b = 0; b < batch; ++b) {
        cb_sync2.reserve_back(1);
#ifdef ENABLE_GLOBAL_CB
        experimental::remote_cb_wait_front(remote_cb_id, num_blocks);
#endif

        cb_sync2.push_back(1);

        if constexpr (in1_is_dram_interleaved) {
            for (uint32_t block = 0; block < num_blocks; ++block) {
                uint32_t block_idx = (ring_idx + block) % num_blocks;

                cb_in1.reserve_back(in1_block_num_tiles);
                read_block_from_dram(
                    noc_obj,
                    cb_in1,
                    s1,
                    in1_tensor_width_in_tiles,
                    ring_idx,
                    block_idx,
                    in1_block_width_in_tiles,
                    in1_block_height_in_tiles,
                    in1_single_tile_size_bytes);
                cb_in1.push_back(in1_block_num_tiles);
            }
        } else if constexpr (in1_is_dram_sharded) {  // when in1 is sharded in DRAM, each core reads from its own bank,
                                                     // two cores on the same row share one bank.
            for (uint32_t block = 0; block < num_blocks; ++block) {
                uint32_t block_idx = (ring_idx + block) % num_blocks;
                l1_read_addr_in1 = block_idx * in1_dram_shard_block_size_bytes + dram_read_offset_bytes;
                // Operand 1
                cb_in1.reserve_back(in1_block_num_tiles);
                l1_write_addr_in1 = cb_in1.get_write_ptr();

                for (uint32_t h = 0; h < in1_block_height_in_tiles; ++h) {
                    uint32_t curr_l1_read_addr_in1 = l1_read_addr_in1;
                    for (uint32_t w = 0; w < in1_block_width_num_pages; ++w) {
                        uint32_t curr_page_size =
                            w == in1_block_width_num_pages - 1 ? in1_block_page_size_last : in1_block_page_size;
                        uint64_t in1_base_addr = get_noc_addr_from_bank_id<true>(dram_bank_id, in1_tensor_addr);
                        noc_async_read_one_packet_set_state<true>(in1_base_addr, curr_page_size, vc);
                        noc_async_read_one_packet_with_state<true, true>(
                            in1_base_addr + curr_l1_read_addr_in1, l1_write_addr_in1, vc);
                        curr_l1_read_addr_in1 += curr_page_size;
                        l1_write_addr_in1 += curr_page_size;
                    }
                    l1_read_addr_in1 += in1_shard_width_offset_bytes;
                }

                noc_obj.async_read_barrier();
                cb_in1.push_back(in1_block_num_tiles);
            }
        }

#ifdef ENABLE_GLOBAL_CB
        cb_sync.wait_front(1);
        experimental::remote_cb_pop_front(remote_cb_id, num_blocks);
        cb_sync.pop_front(1);
#endif
        // Signal Here
        if constexpr (needs_signaler) {
            if (b == 0) {
                do_signaling(noc_obj, rt_args_idx);
            }
        }
    }

#ifdef ENABLE_GLOBAL_CB
    experimental::update_remote_cb_config_in_l1(remote_cb_id);
    noc_obj.async_atomic_barrier();
#endif
    noc_obj.async_write_barrier();
}

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "debug/assert.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_utils.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"

using ttnn::ccl::ShardType;
using ttnn::ccl::UNINITIALIZED_VALUE_U16;
using ttnn::ccl::UNINITIALIZED_VALUE_U32;
using ttnn::ccl::WorkerXY;

// Only workers on local worker core, hence no uint64_t noc addresses
template <ShardType SHARD_TYPE>
struct FullWorkerGridShardAddrGen {
    FullWorkerGridShardAddrGen() = default;
    FORCE_INLINE static void build_with_placement_new(
        FullWorkerGridShardAddrGen* placement_new_address, const uint32_t arg_index) {
        ttnn::ccl::FullWorkerGridShardAddrGenArgs<false> input_args;

        uint32_t curr_arg_index = arg_index;
        input_args.tile_size_in_bytes = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.shards_start_address = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.curr_shard_tile_x = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.curr_shard_tile_y = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.curr_tile_index = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.curr_shard = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.input_shard_num_tiles_x = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.input_shard_num_tiles_y = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.total_shards_x = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.is_clockwise = get_arg_val<uint32_t>(curr_arg_index++) != 0;
        input_args.curr_core_index = static_cast<uint16_t>(get_arg_val<uint32_t>(curr_arg_index++));
        input_args.total_num_cores = static_cast<uint16_t>(get_arg_val<uint32_t>(curr_arg_index++));
        input_args.dest_cores = reinterpret_cast<WorkerXY*>(get_arg_addr(curr_arg_index));
        curr_arg_index += input_args.total_num_cores;

        ASSERT(input_args.tile_size_in_bytes != UNINITIALIZED_VALUE_U32);
        ASSERT(input_args.shards_start_address != UNINITIALIZED_VALUE_U32);
        ASSERT(input_args.curr_core_index != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.total_num_cores != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.curr_shard_tile_x != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.curr_shard_tile_y != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.curr_tile_index != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.curr_shard != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.input_shard_num_tiles_x != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.input_shard_num_tiles_y != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.total_shards_x != UNINITIALIZED_VALUE_U16);

        ASSERT(curr_arg_index - arg_index == input_args.get_expected_num_args());

        new (placement_new_address) FullWorkerGridShardAddrGen(curr_arg_index - arg_index, input_args);
    }

    FullWorkerGridShardAddrGen(
        uint8_t num_args_consumed, ttnn::ccl::FullWorkerGridShardAddrGenArgs<false> const& input_args) :
        dest_cores(input_args.dest_cores),
        tile_size_in_bytes(input_args.tile_size_in_bytes),
        shards_start_address(input_args.shards_start_address),
        curr_core_index(input_args.curr_core_index),
        total_num_cores(input_args.total_num_cores),
        curr_shard_tile_x(input_args.curr_shard_tile_x),
        curr_shard_tile_y(input_args.curr_shard_tile_y),
        curr_tile_index(input_args.curr_tile_index),
        curr_shard(input_args.curr_shard),
        input_shard_num_tiles_x(input_args.input_shard_num_tiles_x),
        input_shard_num_tiles_y(input_args.input_shard_num_tiles_y),
        total_shards_x(input_args.total_shards_x),
        num_args_consumed(num_args_consumed),
        is_clockwise(input_args.is_clockwise) {
        ASSERT(input_shard_num_tiles_x > 0);
        ASSERT(input_shard_num_tiles_y > 0);
        ASSERT(total_shards_x > 0);
        ASSERT(total_num_cores > 0);
        ASSERT(curr_core_index < total_num_cores);
        if constexpr (SHARD_TYPE == ShardType::Width) {
            ASSERT(curr_shard < total_shards_x);
            ASSERT(
                curr_tile_index = curr_shard_tile_x * input_shard_num_tiles_x +
                                  (curr_shard_tile_y * total_shards_x * input_shard_num_tiles_x));
        } else {
            ASSERT(false);  // Not implemented yet
        }
    }

    [[nodiscard]] FORCE_INLINE WorkerXY get_next_noc_xy_core() const {
        ASSERT(this->curr_core_index < this->total_num_cores);
        return this->dest_cores[this->curr_core_index];
    }

    [[nodiscard]] FORCE_INLINE uint64_t get_next_noc_addr() const {
        WorkerXY dest_worker = this->get_next_noc_xy_core();
        uint32_t curr_address = this->shards_start_address + this->curr_tile_index * this->tile_size_in_bytes;
        ASSERT(this->shards_start_address <= curr_address);
        return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
    }

    FORCE_INLINE uint16_t get_tiles_left_in_row_in_shard() const {
        return this->input_shard_num_tiles_x - this->curr_shard_tile_x;
    }

    FORCE_INLINE void advance() {
        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance(
            this->curr_shard_tile_x,
            this->curr_shard_tile_y,
            this->curr_tile_index,
            this->curr_core_index,
            this->total_num_cores,
            this->input_shard_num_tiles_x,
            this->input_shard_num_tiles_y,
            this->total_shards_x,
            this->curr_shard,
            this->is_clockwise);
    }

    FORCE_INLINE void advance_to_next_tile_row() {
        ttnn::ccl::all_gather::full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
            this->curr_shard_tile_x,
            this->curr_shard_tile_y,
            this->curr_tile_index,
            this->curr_core_index,
            this->total_num_cores,
            this->input_shard_num_tiles_x,
            this->input_shard_num_tiles_y,
            this->total_shards_x,
            this->curr_shard,
            this->is_clockwise);
    }

    FORCE_INLINE void advance_n_tiles(uint16_t n) {
        // TODO: optimize
        for (uint16_t i = 0; i < n; i++) {
            this->advance();
        }
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_tile_size_in_bytes() const { return this->tile_size_in_bytes; }

    [[nodiscard]] FORCE_INLINE uint32_t get_shard_tile_row_size_in_bytes() const {
        return this->input_shard_num_tiles_x * this->tile_size_in_bytes;
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_num_args_consumed() const { return this->num_args_consumed; }

    WorkerXY* dest_cores;
    uint32_t tile_size_in_bytes;
    uint32_t shards_start_address;
    uint16_t curr_core_index;
    uint16_t total_num_cores;
    uint16_t curr_shard_tile_x;
    uint16_t curr_shard_tile_y;
    uint16_t curr_tile_index;
    uint16_t curr_shard;
    uint16_t input_shard_num_tiles_x;
    uint16_t input_shard_num_tiles_y;
    uint16_t total_shards_x;
    uint8_t num_args_consumed;
    bool is_clockwise;
};

template <ShardType TYPE>
struct ShardAddrGen final {
    ShardAddrGen() = default;

    FORCE_INLINE static void build_with_placement_new(ShardAddrGen* placement_new_address, const uint32_t arg_index) {
        ttnn::ccl::ShardAddrGenArgs<false> input_args;

        uint32_t curr_arg_index = arg_index;
        input_args.is_clockwise = bool(get_arg_val<uint32_t>(curr_arg_index++) == 1);
        input_args.shard_size_in_bytes = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.total_chunks_per_core = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.shards_start_address = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.starting_core_index = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.starting_chunk_into_shard = get_arg_val<uint32_t>(curr_arg_index++);

        input_args.intra_core_stride_in_shards = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.contiguous_chunks_before_stride = get_arg_val<uint32_t>(curr_arg_index++);

        input_args.num_dest_cores = get_arg_val<uint32_t>(curr_arg_index++);
        input_args.dest_cores = reinterpret_cast<WorkerXY*>(get_arg_addr(curr_arg_index));
        curr_arg_index += input_args.num_dest_cores;

        ASSERT(input_args.shard_size_in_bytes != UNINITIALIZED_VALUE_U32);
        ASSERT(input_args.total_chunks_per_core != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.shards_start_address != UNINITIALIZED_VALUE_U32);
        ASSERT(input_args.starting_core_index != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.starting_chunk_into_shard != UNINITIALIZED_VALUE_U16);
        ASSERT(input_args.num_dest_cores != UNINITIALIZED_VALUE_U16);

        ASSERT(curr_arg_index - arg_index == input_args.get_expected_num_args());

        new (placement_new_address) ShardAddrGen(curr_arg_index - arg_index, input_args);
    }

    // This addr gen will dump all tiles from an input shard contiguously, and dump the
    // next input shard contiguously after it. This approach depends on a follow up
    //
    ShardAddrGen(uint8_t num_args_consumed, ttnn::ccl::ShardAddrGenArgs<false> const& input_args) :
        dest_cores(input_args.dest_cores),
        shards_start_address(input_args.shards_start_address),
        shard_size_in_bytes(input_args.shard_size_in_bytes),
        total_chunks_per_core(input_args.total_chunks_per_core),
        curr_worker_index(input_args.starting_core_index),
        curr_core_chunk_index(input_args.starting_chunk_into_shard),

        intra_core_stride_in_shards(input_args.intra_core_stride_in_shards),
        contiguous_chunk_count(1),
        contiguous_chunks_before_stride(input_args.contiguous_chunks_before_stride),
        num_dest_cores(input_args.num_dest_cores),

        num_args_consumed(num_args_consumed),
        is_clockwise(input_args.is_clockwise) {
        ASSERT(this->contiguous_chunks_before_stride >= 1);
        ASSERT(this->intra_core_stride_in_shards >= 1);
        ASSERT(input_args.starting_chunk_into_shard <= this->total_chunks_per_core);
    };

    static_assert(
        TYPE == ShardType::Width || TYPE == ShardType::Height || TYPE == ShardType::Block, "Invalid ShardType");

    // Clockwise vs counter clockwise only affects worker core traversal order (relative to canonical order). Since the
    // dest core list is a configurable list, we will, for now, require the host side kernel config code to produce the
    // correc order per worker
    FORCE_INLINE void advance() {
        if constexpr (TYPE == ShardType::Width or TYPE == ShardType::Height) {
            ttnn::ccl::all_gather::addr_gen_advance_width_sharded(
                this->curr_core_chunk_index,
                this->curr_worker_index,
                this->contiguous_chunk_count,
                this->total_chunks_per_core,
                this->num_dest_cores,
                this->intra_core_stride_in_shards,
                this->contiguous_chunks_before_stride,
                this->is_clockwise);
        } else {
            // Unsupported
            ASSERT(false);
        }
    }

    [[nodiscard]] FORCE_INLINE WorkerXY get_next_noc_xy_core() const {
        ASSERT(this->curr_worker_index < this->num_dest_cores);
        return this->dest_cores[this->curr_worker_index];
    }

    [[nodiscard]] FORCE_INLINE uint64_t get_next_noc_addr() const {
        WorkerXY dest_worker = this->get_next_noc_xy_core();
        uint32_t curr_address = this->shards_start_address + this->curr_core_chunk_index * this->shard_size_in_bytes;
        ASSERT(this->shards_start_address <= curr_address);
        return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
    }

    [[nodiscard]] FORCE_INLINE uint64_t get_next_noc_addr_and_advance() {
        if constexpr (TYPE == ShardType::Width) {
            WorkerXY dest_worker = this->get_next_noc_xy_core();
            uint32_t curr_address =
                this->shards_start_address + this->curr_core_chunk_index * this->shard_size_in_bytes;
            ASSERT(this->shards_start_address <= curr_address);
            this->advance();
            return get_noc_addr(dest_worker.x, dest_worker.y, curr_address);
        } else {
            ASSERT(false);
            // Unsupported
            return 0;
        }
    }

    [[nodiscard]] FORCE_INLINE uint32_t get_shard_size_in_bytes() const { return this->shard_size_in_bytes; }

    [[nodiscard]] FORCE_INLINE uint32_t get_num_dest_cores() const { return this->num_dest_cores; }
    [[nodiscard]] FORCE_INLINE uint32_t get_total_chunks_per_core() const { return this->total_chunks_per_core; }
    [[nodiscard]] FORCE_INLINE uint32_t get_num_args_consumed() const { return this->num_args_consumed; }

    WorkerXY* dest_cores;
    uint32_t shards_start_address;
    uint32_t shard_size_in_bytes;
    uint16_t total_chunks_per_core;
    uint16_t curr_worker_index;
    uint16_t curr_core_chunk_index;
    uint16_t intra_core_stride_in_shards;
    uint16_t contiguous_chunk_count;
    uint16_t contiguous_chunks_before_stride;
    uint16_t num_dest_cores;
    uint8_t num_args_consumed;
    bool is_clockwise;
};

template <ShardType T>
FORCE_INLINE void write_and_send_chunk_sharded(
    const uint32_t& cb_id,
    ShardAddrGen<T>& addr_gen,
    uint32_t const num_pages,
    uint64_t remote_eth_l1_write_addr,
    uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint32_t num_pages_remaining = num_pages;
    noc_async_write(l1_read_addr, remote_eth_l1_write_addr, num_pages * addr_gen.get_shard_size_in_bytes());
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    while (num_pages_remaining > 0) {
        uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr();
        uint32_t num_shards_to_write =
            std::min<uint32_t>(num_pages_remaining, addr_gen.contiguous_chunks_before_stride);
        noc_async_write(l1_read_addr, dest_worker_noc_addr, num_shards_to_write * addr_gen.get_shard_size_in_bytes());
        for (uint32_t i = 0; i < num_shards_to_write; i++) {
            addr_gen.advance();
        }
        num_pages_remaining -= num_shards_to_write;
        l1_read_addr += num_shards_to_write * addr_gen.get_shard_size_in_bytes();
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template <typename AddrGen>
FORCE_INLINE void write_and_send_chunk(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t num_cols,
    const uint32_t num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size,
    uint64_t remote_l1_write_addr,
    uint64_t eth_l1_sender_semaphore_addr) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    noc_async_write(l1_read_addr, remote_l1_write_addr, page_size * num_pages);
    noc_semaphore_inc(eth_l1_sender_semaphore_addr, 1);
    // TODO: do eth semaphore inc here
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILE_INTERLEAVED || defined SHARDED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void write_chunk_sharded(const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, const uint32_t num_pages) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    uint32_t num_pages_remaining = num_pages;
    while (num_pages_remaining > 0) {
        uint64_t dest_worker_noc_addr = addr_gen.get_next_noc_addr();
        uint32_t num_contiguous_shards = addr_gen.contiguous_chunks_before_stride;
        uint32_t num_to_send = std::min(num_pages_remaining, num_contiguous_shards);
        noc_async_write(l1_read_addr, dest_worker_noc_addr, num_to_send * addr_gen.get_shard_size_in_bytes());
        for (uint32_t i = 0; i < num_to_send; i++) {
            addr_gen.advance();
        }
        l1_read_addr += num_to_send * addr_gen.get_shard_size_in_bytes();
        num_pages_remaining -= num_to_send;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}
template <typename AddrGen>
FORCE_INLINE void write_chunk(
    uint32_t& output_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& d,
    const uint32_t& num_cols,
    const uint32_t& num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(output_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        output_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            output_page_idx += row_offset;
        }
#elif defined TILE_INTERLEAVED
        noc_async_write_tile(output_page_idx, d, l1_read_addr);
        output_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            output_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                output_page_idx += row_offset;
            }
        }
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <ShardType T>
FORCE_INLINE void read_shard_from_input_tensor_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t num_shards) {
    cb_reserve_back(cb_id, num_shards);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    for (uint32_t s = 0; s < num_shards; s++) {
        uint64_t src_noc_addr = addr_gen.get_next_noc_addr_and_advance();
        noc_async_read(src_noc_addr, local_l1_read_dest_addr, addr_gen.get_shard_size_in_bytes());
        local_l1_read_dest_addr += addr_gen.get_shard_size_in_bytes();
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_shards);
}
// read chunk from input tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_input_tensor(
    uint32_t& input_page_idx,
    const uint32_t& cb_id,
    const AddrGen& s,
    const uint32_t& num_pages,
    const uint32_t& page_size) {
    const uint32_t end_read_idx = input_page_idx + num_pages;
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (; input_page_idx < end_read_idx; ++input_page_idx) {
#ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
#elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
#endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

// Same function - just different address generators? Commonize later
template <ShardType T>
FORCE_INLINE void read_chunk_from_output_tensor_sharded(
    const uint32_t& cb_id, ShardAddrGen<T>& addr_gen, uint32_t const num_pages) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_dest_addr = get_write_ptr(cb_id);
    uint32_t num_pages_remaining = num_pages;
    while (num_pages_remaining > 0) {
        uint64_t src_noc_addr = addr_gen.get_next_noc_addr();
        uint32_t shards_to_read = std::min<uint32_t>(num_pages_remaining, addr_gen.contiguous_chunks_before_stride);
        noc_async_read(src_noc_addr, local_l1_read_dest_addr, shards_to_read * addr_gen.get_shard_size_in_bytes());
        local_l1_read_dest_addr += shards_to_read * addr_gen.get_shard_size_in_bytes();
        for (uint32_t i = 0; i < shards_to_read; i++) {
            addr_gen.advance();
        }
        num_pages_remaining -= shards_to_read;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}
// read chunk from output tensor (local chip)
template <typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor(
    uint32_t& input_page_idx,
    uint32_t& col_idx,
    uint32_t& row_idx,
    const uint32_t& cb_id,
    const AddrGen& s,
    const uint32_t& num_cols,
    const uint32_t& num_rows,
    const uint32_t& col_offset,
    const uint32_t& row_offset,
    const uint32_t& num_pages,
    const uint32_t& page_size) {
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(input_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        input_page_idx++;
        row_idx++;
        if (row_idx == num_rows) {
            row_idx = 0;
            input_page_idx += row_offset;
        }
#elif defined TILE_INTERLEAVED
        noc_async_read_tile(input_page_idx, s, local_l1_read_addr);
        input_page_idx++;
        col_idx++;
        if (col_idx == num_cols) {
            input_page_idx += col_offset;
            col_idx = 0;
            row_idx++;
            if (row_idx == num_rows) {
                row_idx = 0;
                input_page_idx += row_offset;
            }
        }
#endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void read_chunk_from_output_tensor_v2(
    uint32_t& curr_page_idx,
    ttnn::ccl::coord_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    const uint32_t cb_id,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {
    // we expected caller to reset this and the last curr_page_idx when we set it true
    ASSERT(last_page_of_worker == false);
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(curr_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        ASSERT(false);  // unimplemented

#elif defined TILE_INTERLEAVED

        noc_async_read_tile(curr_page_idx, s, local_l1_read_addr);
        // common with `write_chunk_v2`
        offset_into_worker_slice.x++;
        bool end_of_worker_slice_row = offset_into_worker_slice.x == worker_slice_shape.x;
        if (end_of_worker_slice_row) {
            offset_into_worker_slice.x = 0;
            offset_into_worker_slice.y++;
            bool end_of_worker_slice = offset_into_worker_slice.y == worker_slice_shape.y;
            if (end_of_worker_slice) {
                offset_into_worker_slice.y = 0;
                last_page_of_worker = true;
            } else {
                curr_page_idx += tensor_shape.x - worker_slice_shape.x;
            }
        } else {
            curr_page_idx++;
        }
#endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void write_chunk_v2(
    uint32_t& curr_page_idx,
    ttnn::ccl::coord_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    uint32_t cb_id,
    const AddrGen& d,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {
    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(curr_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        ASSERT(false);  // unimplemented
#elif defined TILE_INTERLEAVED
        noc_async_write_tile(curr_page_idx, d, l1_read_addr);
        // Common with `read_chunk_from_output_tensor_v2`
        offset_into_worker_slice.x++;
        bool end_of_worker_slice_row = offset_into_worker_slice.x == worker_slice_shape.x;
        if (end_of_worker_slice_row) {
            offset_into_worker_slice.x = 0;
            offset_into_worker_slice.y++;
            bool end_of_worker_slice = offset_into_worker_slice.y == worker_slice_shape.y;
            if (end_of_worker_slice) {
                offset_into_worker_slice.y = 0;
                last_page_of_worker = true;
            } else {
                curr_page_idx += tensor_shape.x - worker_slice_shape.x;
            }
        } else {
            curr_page_idx++;
        }
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void read_wrapped_chunk_from_output_tensor(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
     ttnn::ccl::coord_t& offset_worker_slice,
    const  ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const  ttnn::ccl::coord_t& tensor_shape,
    const  ttnn::ccl::coord_t& tensor_slice_shape,
    const uint32_t cb_id,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {

    // we expected caller to reset this and the last curr_page_idx when we set it true
    ASSERT(last_page_of_worker == false);
    cb_reserve_back(cb_id, num_pages);
    uint32_t local_l1_read_addr = get_write_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t src_noc_addr = get_noc_addr(curr_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
        ASSERT(false);  // unimplemented

#elif defined TILE_INTERLEAVED

        noc_async_read_tile(curr_page_idx, s, local_l1_read_addr);
        // common with `write_chunk_v2`

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx, // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            last_page_of_worker
        );

#endif
        local_l1_read_addr += page_size;
    }
    noc_async_read_barrier();
    cb_push_back(cb_id, num_pages);
}

template <typename AddrGen>
FORCE_INLINE void write_wrapped_chunk(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
     ttnn::ccl::coord_t& offset_worker_slice,
    const  ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const  ttnn::ccl::coord_t& tensor_shape,
    const  ttnn::ccl::coord_t& tensor_slice_shape,
    uint32_t cb_id,
    const AddrGen& d,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {

    cb_wait_front(cb_id, num_pages);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = 0; i < num_pages; ++i) {
#ifdef RM_INTERLEAVED
        uint64_t dst_noc_addr = get_noc_addr(curr_page_idx, d);
        noc_async_write(l1_read_addr, dst_noc_addr, page_size);
        ASSERT(false);  // unimplemented
#elif defined TILE_INTERLEAVED
        noc_async_write_tile(curr_page_idx, d, l1_read_addr);
        // Common with `read_chunk_from_output_tensor_v2`

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx, // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            last_page_of_worker
        );
#endif
        l1_read_addr += page_size;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id, num_pages);
}

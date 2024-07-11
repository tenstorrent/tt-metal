// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
// #include <type_traits>
#include <limits>
#include <vector>

namespace ttnn {
namespace utils {
namespace ccl {

enum EriscDataMoverBufferSharingMode : uint32_t {
    NOT_SHARED = 0,
    ROUND_ROBIN = 1,
    SHARED = 2,
    ROUND_ROBIN_AND_SHARED = 3
};

enum EriscDataMoverTerminationMode : uint32_t { MESSAGE_COUNT_REACHED = 0, WORKER_INITIATED = 1 };

enum EriscDataMoverWorkerSignal : uint32_t {
    NEXT_MESSAGE_AVAILABLE = 1,
    NEXT_MESSAGE_IS_LAST = 2,
    TERMINATE_IMMEDIATELY = 3,
};

// TODO: let the kernel runtime args

enum ShardType : uint8_t { Width = 0, Height = 1, Block = 2 };

/*
 * Worker coordinate, used by EDM and some kernels to know which workers to signal
 */
struct WorkerXY {
    uint16_t x;
    uint16_t y;

    WorkerXY(uint16_t x, uint16_t y) : x(x), y(y) {}

    uint32_t to_uint32() const { return (y << 16) | x; }

    bool operator==(const WorkerXY &rhs) const { return x == rhs.x && y == rhs.y; }
    bool operator!=(const WorkerXY &rhs) const { return !(*this == rhs); }
};

struct coord_t {
    coord_t(uint32_t x, uint32_t y) : x(x), y(y) {}
    uint32_t x;
    uint32_t y;
};

// Advances relative to immediate outer slice. There is no notion of global offset here and the caller would be expected
// to add any additional offsets required. Consider templatizing this to conditionally implement the divide as a shift
inline coord_t advance_slice_row_major(
    coord_t const &inner_slice_offset,
    coord_t const &inner_slice_shape,
    coord_t const &outer_slice_shape,
    uint32_t num_active_slices) {
    auto slice_mod_x = outer_slice_shape.x % inner_slice_shape.x;
    bool needs_padding = slice_mod_x != 0;
    coord_t padded_outer_slice_shape =
        needs_padding ? coord_t(outer_slice_shape.x + (inner_slice_shape.x - slice_mod_x), outer_slice_shape.y)
                      : outer_slice_shape;
    uint32_t advance_x = inner_slice_shape.x * num_active_slices;
    uint32_t next_offset_x = inner_slice_offset.x + advance_x;
    if (next_offset_x < padded_outer_slice_shape.x) {
        return coord_t(next_offset_x, inner_slice_offset.y);
    }

    uint32_t advance_x_from_0_offset_x = next_offset_x - padded_outer_slice_shape.x;
    uint32_t next_offset_y = inner_slice_offset.y + inner_slice_shape.y;
    // Measure perf impact of early exit vs the division
    if (advance_x_from_0_offset_x < padded_outer_slice_shape.x) {
        return coord_t(advance_x_from_0_offset_x, inner_slice_offset.y + inner_slice_shape.y);
    }

    uint32_t slice_rows_advanced = advance_x_from_0_offset_x / padded_outer_slice_shape.x;
    next_offset_x = advance_x_from_0_offset_x - (slice_rows_advanced * padded_outer_slice_shape.x);
    next_offset_y += slice_rows_advanced * inner_slice_shape.y;

    return coord_t(next_offset_x, next_offset_y);
}

static constexpr uint32_t UNINITIALIZED_VALUE_U32 = std::numeric_limits<uint32_t>::max();
static constexpr uint16_t UNINITIALIZED_VALUE_U16 = std::numeric_limits<uint16_t>::max();

template <bool T>
struct ArchDependentTypes;

template <>
struct ArchDependentTypes<true> {
    using workers_list_t = std::vector<ccl::WorkerXY>;
    static const workers_list_t WORKERS_LIST_UNINITIALIZED_VALUE;
};

template <>
struct ArchDependentTypes<false> {
    using workers_list_t = ttnn::utils::ccl::WorkerXY *;
    static const workers_list_t WORKERS_LIST_UNINITIALIZED_VALUE;
};

template <bool IS_HOST>
struct FullWorkerGridShardAddrGenArgs final {
    typename ArchDependentTypes<IS_HOST>::workers_list_t dest_cores;
    uint32_t tile_size_in_bytes = UNINITIALIZED_VALUE_U32;
    uint32_t shards_start_address = UNINITIALIZED_VALUE_U32;
    uint16_t curr_core_index = UNINITIALIZED_VALUE_U16;
    uint16_t total_num_cores = UNINITIALIZED_VALUE_U16;
    uint16_t curr_shard_tile_x = UNINITIALIZED_VALUE_U16;
    uint16_t curr_shard_tile_y = UNINITIALIZED_VALUE_U16;
    uint16_t curr_tile_index = UNINITIALIZED_VALUE_U16;
    uint16_t curr_shard = UNINITIALIZED_VALUE_U16;
    uint16_t input_shard_num_tiles_x = UNINITIALIZED_VALUE_U16;
    uint16_t input_shard_num_tiles_y = UNINITIALIZED_VALUE_U16;
    uint16_t total_shards_x = UNINITIALIZED_VALUE_U16;
    bool is_clockwise = false;

    inline uint32_t get_expected_num_args() const {
        if constexpr (IS_HOST) {
            return 12 + this->total_num_cores;
        } else {
            return 12 + this->total_num_cores;
        }
    }
};

template <bool IS_HOST>
struct ShardAddrGenArgs final {
    uint32_t shards_start_address = UNINITIALIZED_VALUE_U32;
    uint32_t shard_size_in_bytes = UNINITIALIZED_VALUE_U32;
    uint16_t total_chunks_per_core = UNINITIALIZED_VALUE_U16;

    uint16_t starting_core_index = UNINITIALIZED_VALUE_U16;
    uint16_t starting_chunk_into_shard = UNINITIALIZED_VALUE_U16;

    uint16_t intra_core_stride_in_shards = UNINITIALIZED_VALUE_U16;
    uint16_t contiguous_chunks_before_stride = UNINITIALIZED_VALUE_U16;

    uint16_t num_dest_cores = UNINITIALIZED_VALUE_U16;

    typename ArchDependentTypes<IS_HOST>::workers_list_t dest_cores;
    bool is_clockwise = false;

    inline uint32_t get_expected_num_args() const {
        if constexpr (IS_HOST) {
            return 9 + dest_cores.size();
        } else {
            return 9 + this->num_dest_cores;
        }
    }
};

namespace all_gather {
inline void addr_gen_advance_width_sharded(
    uint16_t &curr_core_tile_index,
    uint16_t &curr_worker_index,
    uint16_t &contiguous_tile_count,
    // uint16_t& current_core_chunks_visited,
    const uint16_t &total_chunks_per_core,
    const uint16_t &num_dest_cores,
    const uint16_t &intra_core_stride_in_shards,
    const uint16_t &contiguous_chunks_before_stride,
    bool is_clockwise) {
    if (is_clockwise) {
        bool do_stride = contiguous_tile_count == contiguous_chunks_before_stride;
        bool stride_induced_chunk_wraparound =
            (do_stride && curr_core_tile_index < (intra_core_stride_in_shards + contiguous_chunks_before_stride - 1));
        bool do_chunk_wrap = curr_core_tile_index >= total_chunks_per_core || stride_induced_chunk_wraparound;

        if (do_chunk_wrap) {
            bool do_core_wrap = curr_worker_index == 0;
            uint32_t past_end_index =
                (total_chunks_per_core + curr_core_tile_index + 1 - contiguous_chunks_before_stride);
            uint32_t backward_step_amount = (intra_core_stride_in_shards + contiguous_chunks_before_stride - 1);
            curr_core_tile_index = past_end_index - backward_step_amount;
            contiguous_tile_count = 1;
            if (do_core_wrap) {
                curr_worker_index = num_dest_cores - 1;
            } else {
                curr_worker_index--;
            }
        } else {
            if (do_stride) {
                contiguous_tile_count = 1;
                curr_core_tile_index -= (intra_core_stride_in_shards + contiguous_chunks_before_stride - 1);
            } else {
                contiguous_tile_count++;
                curr_core_tile_index++;
            }
        }

    } else {
        if (contiguous_tile_count == contiguous_chunks_before_stride) {
            contiguous_tile_count = 1;
            curr_core_tile_index += intra_core_stride_in_shards;
        } else {
            contiguous_tile_count++;
            curr_core_tile_index++;
        }

        bool do_chunk_wrap = curr_core_tile_index >= total_chunks_per_core;
        if (do_chunk_wrap) {
            curr_core_tile_index = curr_core_tile_index - total_chunks_per_core;
            curr_worker_index++;
            bool do_core_wrap = curr_worker_index == num_dest_cores;
            if (do_core_wrap) {
                curr_worker_index = 0;
            }
        }
    }
}

inline void full_worker_grid_addr_gen_width_sharded_advance_shard_impl(
    uint16_t &curr_shard_tile_x,
    uint16_t &curr_shard_tile_y,
    uint16_t &curr_tile_index,
    uint16_t &curr_core_index,
    uint16_t const total_num_cores,
    uint16_t const input_shard_num_tiles_x,
    uint16_t const total_shards_x,
    uint16_t const shard_offset,
    bool is_clockwise) {
    bool wrap_around = is_clockwise ? curr_core_index == 0 : curr_core_index == total_num_cores - 1;
    curr_core_index = wrap_around ? (is_clockwise ? total_num_cores - 1 : 0)
                                  : (is_clockwise ? curr_core_index - 1 : curr_core_index + 1);
    curr_tile_index = input_shard_num_tiles_x * shard_offset;
    curr_shard_tile_x = 0;
    curr_shard_tile_y = 0;
}

// TODO: support cases where the full shard is contiguous (e.g. height sharded)
inline void full_worker_grid_addr_gen_width_sharded_advance_full_tile_row(
    uint16_t &curr_shard_tile_x,
    uint16_t &curr_shard_tile_y,
    uint16_t &curr_tile_index,
    uint16_t &curr_core_index,
    uint16_t const total_num_cores,
    uint16_t const input_shard_num_tiles_x,
    uint16_t const input_shard_num_tiles_y,
    uint16_t const total_shards_x,
    uint16_t const shard_offset,
    bool is_clockwise) {
    // Keep it verbose for now. we can reduce to a flat index later
    bool is_last_row = curr_shard_tile_y == input_shard_num_tiles_y - 1;
    if (is_last_row) {
        full_worker_grid_addr_gen_width_sharded_advance_shard_impl(
            curr_shard_tile_x,
            curr_shard_tile_y,
            curr_tile_index,
            curr_core_index,
            total_num_cores,
            input_shard_num_tiles_x,
            total_shards_x,
            shard_offset,
            is_clockwise);

    } else {
        curr_tile_index += total_shards_x * input_shard_num_tiles_x - curr_shard_tile_x;
        curr_shard_tile_x = 0;
        curr_shard_tile_y++;
    }
}

inline void full_worker_grid_addr_gen_width_sharded_advance(
    uint16_t &curr_shard_tile_x,
    uint16_t &curr_shard_tile_y,
    uint16_t &curr_tile_index,
    uint16_t &curr_core_index,
    uint16_t const total_num_cores,
    uint16_t const input_shard_num_tiles_x,
    uint16_t const input_shard_num_tiles_y,
    uint16_t const total_shards_x,
    uint16_t const shard_offset,
    bool is_clockwise) {
    // Keep it verbose for now. we can reduce to a flat index later
    bool last_tile_in_row = curr_shard_tile_x == input_shard_num_tiles_x - 1;
    bool last_tile_in_col = curr_shard_tile_y == input_shard_num_tiles_y - 1;
    if (last_tile_in_row && last_tile_in_col) {
        full_worker_grid_addr_gen_width_sharded_advance_shard_impl(
            curr_shard_tile_x,
            curr_shard_tile_y,
            curr_tile_index,
            curr_core_index,
            total_num_cores,
            input_shard_num_tiles_x,
            total_shards_x,
            shard_offset,
            is_clockwise);

    } else if (last_tile_in_row) {
        curr_tile_index += total_shards_x * input_shard_num_tiles_x - curr_shard_tile_x;
        curr_shard_tile_x = 0;
        curr_shard_tile_y++;
    } else {
        curr_shard_tile_x++;
        curr_tile_index++;
    }
}

};  // namespace all_gather

}  // namespace ccl
}  // namespacett::tt_metal
}  // namespace tt

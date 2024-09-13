// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
// #include <type_traits>
#include <limits>
#include <vector>

/*
 *    ------   ATTENTION  ATTENTION  ATTENTION  ATTENTION  ATTENTION   ------
 * This file is intended to be useable across both host and device code. Therefore.
 *
 * DO NOT include any headers that are not host/device agnostic.
 * DO NOT use any types that do not have fixed sizes across host and device.
 * e.g. int32_t -> good (always 32 bits), int -> bad (size depends on platform)
 *
 * The reason for dual inclusion across host/device is because this code is used
 * on device, but is further tested on host through gtests. This enables us to
 * sweep functionality quickly and easily without involving end-to-end device kernel
 * invocations and program creation.
 */

namespace ttnn {
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

    constexpr WorkerXY(uint16_t x, uint16_t y) : x(x), y(y) {}

    constexpr uint32_t to_uint32() const { return (y << 16) | x; }

    constexpr bool operator==(const WorkerXY &rhs) const { return x == rhs.x && y == rhs.y; }
    constexpr bool operator!=(const WorkerXY &rhs) const { return !(*this == rhs); }
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

inline coord_t advance_wrapped_slice_row_major(
    coord_t const &inner_slice_offset,
    coord_t const &inner_slice_shape,
    coord_t const &outer_slice_shape,
    uint32_t num_active_slices) {

    uint32_t flattened_inner_slice_offset = inner_slice_offset.x + (inner_slice_offset.y * outer_slice_shape.x);

    uint32_t next_flattened_offset = flattened_inner_slice_offset + (inner_slice_shape.x * inner_slice_shape.y * num_active_slices); // num_active_slices is the total number of workers, so need to stride by that.

    uint32_t next_offset_x = next_flattened_offset % outer_slice_shape.x;
    uint32_t next_offset_y = next_flattened_offset / outer_slice_shape.x;

    return coord_t(next_offset_x, next_offset_y);
}


// Increments the index into the input (global) tensor, while respecting the tensor slice, for wrapped worker slice
// that is internal to the tensor slice.
inline void advance_worker_global_page_interleaved (
    uint32_t &curr_page_idx,
    uint32_t &offset_into_worker_slice, // local to the worker chunk
    coord_t &offset_worker_slice, // local to the tensor slice

    coord_t const &worker_slice_shape, // worker chunk shape
    coord_t const &tensor_slice_shape, // tensor slice shape (per device)

    coord_t const &tensor_shape, // full tensor shape

    const uint32_t stride,
    bool &last_page_of_worker
  ) {

    uint32_t prev_offset_into_worker_slice = offset_into_worker_slice;
    offset_into_worker_slice += stride;

    uint32_t flattened_offset_worker_slice = offset_worker_slice.x + (offset_worker_slice.y * tensor_slice_shape.x);

    // Calculate the number of wrap arounds (cast to uint32_t to **round down**)
    uint32_t prev_num_wrap_around = (flattened_offset_worker_slice + prev_offset_into_worker_slice) / tensor_slice_shape.x;
    uint32_t curr_num_wrap_around = (flattened_offset_worker_slice + offset_into_worker_slice) / tensor_slice_shape.x;
    uint32_t num_wrap_around = curr_num_wrap_around - prev_num_wrap_around;

    bool end_of_worker_slice_row = offset_into_worker_slice == worker_slice_shape.x * worker_slice_shape.y;
    if (end_of_worker_slice_row) {
        offset_into_worker_slice = 0;
        last_page_of_worker = true;
    } else {
        // Check for wrap around
        if (num_wrap_around > 0) { // wrap around wrt to global tensor
            curr_page_idx += num_wrap_around * (tensor_shape.x - tensor_slice_shape.x) + stride;
        } else {
            curr_page_idx += stride;
        }
    }

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
    using workers_list_t = ccl::WorkerXY *;
    static const workers_list_t WORKERS_LIST_UNINITIALIZED_VALUE;
};


}  // namespace ccl
}  // namespace ttnn

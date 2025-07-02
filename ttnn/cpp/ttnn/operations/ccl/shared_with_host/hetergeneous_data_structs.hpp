// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
// #include <type_traits>
#include <limits>
#include <vector>
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"

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
    static constexpr WorkerXY from_uint32(uint32_t v) { return WorkerXY(v & 0xFFFF, (v >> 16) & 0xFFFF); }

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

namespace v2 {
inline size_t flattened_index (const ttnn::ccl::Shape4D<uint32_t>& shape, const ttnn::ccl::Shape4D<uint32_t>& index) {
    std::size_t offset = index.x;
    std::size_t inner_volume = shape.x;
    offset += index.y * inner_volume;
    inner_volume *= shape.y;
    offset += index.z * inner_volume;
    inner_volume *= shape.z;
    offset += index.w * inner_volume;
    return offset;
}

// Increments the index into the input (global) tensor, while respecting the tensor slice, for wrapped worker slice
// that is internal to the tensor slice.
[[nodiscard]] inline bool advance_worker_global_page (
    uint32_t &curr_page_idx,
    uint32_t &offset_into_worker_slice, // local to the worker chunk
    ttnn::ccl::Shape4D<uint32_t> const& offset_worker_slice, // local to the tensor slice

    ttnn::ccl::Shape4D<uint32_t> const &worker_slice_shape, // worker chunk shape
    ttnn::ccl::Shape4D<uint32_t> const &tensor_slice_shape, // tensor slice shape (per device)

    ttnn::ccl::Shape4D<uint32_t> const &tensor_shape, // full tensor shape

    const uint32_t stride
  ) {
    bool may_wrap_multiple_times = stride > tensor_slice_shape.x;
    bool outer_dims_gt_1 = tensor_slice_shape.z > 1 || tensor_slice_shape.w > 1;
    bool end_of_worker_slice_row = false;
    auto next_offset_into_worker_slice = curr_page_idx + stride;
    end_of_worker_slice_row = next_offset_into_worker_slice == worker_slice_shape.volume();
    if (may_wrap_multiple_times || !outer_dims_gt_1) {

        uint32_t prev_offset_into_worker_slice = offset_into_worker_slice;

        uint32_t flattened_offset_worker_slice = flattened_index(tensor_slice_shape, offset_worker_slice);

        // Calculate the number of wrap arounds (cast to uint32_t to **round down**)
        uint32_t prev_num_wrap_around = (flattened_offset_worker_slice + prev_offset_into_worker_slice) / tensor_slice_shape.x;
        uint32_t curr_num_wrap_around = (flattened_offset_worker_slice + next_offset_into_worker_slice) / tensor_slice_shape.x;
        uint32_t num_wrap_around = curr_num_wrap_around - prev_num_wrap_around;

        // Check for wrap around
        if (num_wrap_around > 0) { // wrap around wrt to global tensor
            curr_page_idx += num_wrap_around * (tensor_shape.x - tensor_slice_shape.x) + stride;
        } else {
            curr_page_idx += stride;
        }

    } else {
        // can wrap at-most one time. For now since we only have the flat index, we are going to brute force
        // it. Future work to optimize this - a lot can be done:
        // 1) Carry around the 4D index and also carry around subvolumes
        // 2) Precompute the "inner"/"outer" volumes for each dimension so they are precomputed - this will save
        //    on 4 sums + multiplies per call
        // 3) possibly update address-generators to support n-dimensional indices which may further reduce the number
        //    of operations required (otherwise we still need to eventually convert between flat and)
        // of each dimension so we can more quickly do the striding

        size_t y_x = tensor_slice_shape.y * tensor_slice_shape.x;
        size_t z_y_x = tensor_slice_shape.z * y_x;

        // Calculate the 4D coordinates
        size_t index = next_offset_into_worker_slice;
        size_t new_w = index / z_y_x;
        index -= new_w * z_y_x;

        size_t new_z = index / y_x;
        index -= new_z * y_x;

        size_t new_y = index / tensor_slice_shape.x;
        size_t new_x = index - new_y * tensor_slice_shape.x;

        curr_page_idx = flattened_index(tensor_shape, tensor_slice_shape + Shape4D<uint32_t>{new_x, new_y, new_z, new_w});
    }

    return end_of_worker_slice_row;
}
[[nodiscard]] inline bool advance_worker_global_page (
    uint32_t &curr_page_idx,
    uint32_t &offset_into_worker_slice, // local to the worker chunk
    ttnn::ccl::Shape4D<uint32_t> const& offset_worker_slice, // local to the tensor slice

    size_t const worker_slice_volume, // worker chunk shape
    ttnn::ccl::Shape4D<uint32_t> const &tensor_slice_shape, // tensor slice shape (per device)
    ttnn::ccl::Shape4D<uint32_t> const &tensor_slice_base_offset, // tensor slice shape (per device)

    ttnn::ccl::Shape4D<uint32_t> const &tensor_shape, // full tensor shape

    const uint32_t stride
  ) {
    bool may_wrap_multiple_times = stride > tensor_slice_shape.x;
    bool outer_dims_gt_1 = tensor_slice_shape.z > 1 || tensor_slice_shape.w > 1;
    bool end_of_worker_slice_row = false;
    auto next_offset_into_worker_slice = offset_into_worker_slice + stride;
    end_of_worker_slice_row = next_offset_into_worker_slice == worker_slice_volume;
    if (may_wrap_multiple_times || !outer_dims_gt_1) {
        uint32_t prev_offset_into_worker_slice = offset_into_worker_slice;
        offset_into_worker_slice += stride;

        uint32_t flattened_offset_worker_slice = flattened_index(tensor_slice_shape, offset_worker_slice);

        // Calculate the number of wrap arounds (cast to uint32_t to **round down**)
        uint32_t prev_num_wrap_around = (flattened_offset_worker_slice + prev_offset_into_worker_slice) / tensor_slice_shape.x;
        uint32_t curr_num_wrap_around = (flattened_offset_worker_slice + offset_into_worker_slice) / tensor_slice_shape.x;
        uint32_t num_wrap_around = curr_num_wrap_around - prev_num_wrap_around;

        // Check for wrap around
        if (num_wrap_around > 0) { // wrap around wrt to global tensor
            curr_page_idx += num_wrap_around * (tensor_shape.x - tensor_slice_shape.x) + stride;
        } else {
            curr_page_idx += stride;
        }
    } else {
        // can wrap at-most one time. For now since we only have the flat index, we are going to brute force
        // it. Future work to optimize this - a lot can be done:
        // 1) Carry around the 4D index and also carry around subvolumes
        // 2) Precompute the "inner"/"outer" volumes for each dimension so they are precomputed - this will save
        //    on 4 sums + multiplies per call
        // 3) possibly update address-generators to support n-dimensional indices which may further reduce the number
        //    of operations required (otherwise we still need to eventually convert between flat and)
        // of each dimension so we can more quickly do the striding

        offset_into_worker_slice += stride;
        uint32_t y_x = tensor_slice_shape.y * tensor_slice_shape.x;
        uint32_t z_y_x = tensor_slice_shape.z * y_x;

        // Calculate the 4D coordinates
        uint32_t index = next_offset_into_worker_slice;
        uint32_t new_w = index / z_y_x;
        index -= new_w * z_y_x;

        uint32_t new_z = index / y_x;
        index -= new_z * y_x;

        uint32_t new_y = index / tensor_slice_shape.x;
        uint32_t new_x = index - new_y * tensor_slice_shape.x;

        curr_page_idx = flattened_index(tensor_shape, tensor_slice_base_offset + Shape4D<uint32_t>{new_w, new_z, new_y, new_x});
    }
    return end_of_worker_slice_row;
}

}
// Increments the index into the input (global) tensor, while respecting the tensor slice, for wrapped worker slice
// that is internal to the tensor slice.
inline void advance_worker_global_page_interleaved (
    uint32_t &curr_page_idx,
    uint32_t &offset_into_worker_slice, // local to the worker chunk
    coord_t const& offset_worker_slice, // local to the tensor slice

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

}  // namespace ccl
}  // namespace ttnn

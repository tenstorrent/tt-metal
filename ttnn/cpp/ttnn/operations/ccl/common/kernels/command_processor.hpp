// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_device.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command_device.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"

#include "api/ttnn/tensor/enum_types.hpp"

#include "dataflow_api.h"  // for interleaved addrgen
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/common/interpreter_backends/kernel_common/algorithms.hpp"

using shape_t = ttnn::ccl::Shape4D<uint32_t>;
using address_t = uint32_t;

using tt::tt_metal::BufferType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;
using ttnn::ccl::Shape4D;

#ifdef DEBUG_PRINT_ENABLED
#include "debug/dprint.h"

void dprint(ttnn::ccl::cmd::CclCommandTensor const& command_tensor) {
    DPRINT << "\ttensor_shape.w: " << (uint32_t)command_tensor.tensor_shape.w << "\n";
    DPRINT << "\ttensor_shape.z: " << (uint32_t)command_tensor.tensor_shape.z << "\n";
    DPRINT << "\ttensor_shape.y: " << (uint32_t)command_tensor.tensor_shape.y << "\n";
    DPRINT << "\ttensor_shape.x: " << (uint32_t)command_tensor.tensor_shape.x << "\n";
    DPRINT << "\ttensor_slice_shape.w: " << (uint32_t)command_tensor.tensor_slice_shape.w << "\n";
    DPRINT << "\ttensor_slice_shape.z: " << (uint32_t)command_tensor.tensor_slice_shape.z << "\n";
    DPRINT << "\ttensor_slice_shape.y: " << (uint32_t)command_tensor.tensor_slice_shape.y << "\n";
    DPRINT << "\ttensor_slice_shape.x: " << (uint32_t)command_tensor.tensor_slice_shape.x << "\n";
    DPRINT << "\ttensor_slice_offset.w: " << (uint32_t)command_tensor.tensor_slice_offset.w << "\n";
    DPRINT << "\ttensor_slice_offset.z: " << (uint32_t)command_tensor.tensor_slice_offset.z << "\n";
    DPRINT << "\ttensor_slice_offset.y: " << (uint32_t)command_tensor.tensor_slice_offset.y << "\n";
    DPRINT << "\ttensor_slice_offset.x: " << (uint32_t)command_tensor.tensor_slice_offset.x << "\n";
    DPRINT << "\tworker_start_offset_in_slice.w: " << (uint32_t)command_tensor.worker_start_offset_in_slice.w << "\n";
    DPRINT << "\tworker_start_offset_in_slice.z: " << (uint32_t)command_tensor.worker_start_offset_in_slice.z << "\n";
    DPRINT << "\tworker_start_offset_in_slice.y: " << (uint32_t)command_tensor.worker_start_offset_in_slice.y << "\n";
    DPRINT << "\tworker_start_offset_in_slice.x: " << (uint32_t)command_tensor.worker_start_offset_in_slice.x << "\n";
    DPRINT << "\tworker_pages_per_slice: " << (uint32_t)command_tensor.worker_pages_per_slice << "\n";
}
#endif

void print_tensor_command(uint32_t command_index, ttnn::ccl::cmd::CclCommandTensor const& command_tensor) {
#ifdef DEBUG_PRINT_ENABLED
    DPRINT << "cmd[" << (uint32_t)command_index << "]:\n";
    dprint(command_tensor);
#endif
}

/*
 * Convert a flattened worker offset coord value (assumed 0,0,0, worker offset in pages into tensor slice)
 * into a 4D coordinate value
 */
FORCE_INLINE shape_t worker_wrapped_offset_to_coord(shape_t const& slice_shape, shape_t const& worker_slice_offset) {
    static_assert(
        sizeof(ttnn::ccl::coord_t) == 2 * sizeof(uint32_t), "worker_wrapped_offset_to_coord not updated to work with 4d shape");
    auto const y = worker_slice_offset.x / slice_shape.x;
    return shape_t(0, 0, y, worker_slice_offset.x - (y * slice_shape.x));
}



namespace v2 {
/*
 * Convert a flattened worker offset coord value (assumed 0,0,0, worker offset in pages into tensor slice)
 * into a 4D coordinate value
 */
FORCE_INLINE shape_t worker_wrapped_offset_to_coord(shape_t const& slice_shape, shape_t const& worker_slice_offset) {
    static_assert(
        sizeof(ttnn::ccl::coord_t) == 2 * sizeof(uint32_t), "worker_wrapped_offset_to_coord not updated to work with 4d shape");
    auto const y = worker_slice_offset.x / slice_shape.x;
    return shape_t(0, 0, y, worker_slice_offset.x - (y * slice_shape.x));
}

}  // namespace v2

template <TensorMemoryLayout tensor_layout, tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen {
    static constexpr char name[] = "Uninitialized";
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::INTERLEAVED, buffer_type, page_layout> {
    static constexpr bool is_dram = buffer_type == tt::tt_metal::BufferType::DRAM;
    static constexpr char name[] = "InterleavedAddrGen(default)";
    using type = InterleavedAddrGen<is_dram>;
};
template <tt::tt_metal::BufferType buffer_type>
struct source_tensor_addrgen<TensorMemoryLayout::INTERLEAVED, buffer_type, tt::tt_metal::Layout::TILE> {
    static constexpr bool is_dram = buffer_type == tt::tt_metal::BufferType::DRAM;
    static constexpr char name[] = "InterleavedAddrGen(Tile)";
    using type = InterleavedAddrGenFast<is_dram>;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::WIDTH_SHARDED, buffer_type, page_layout> {
    static constexpr char name[] = "WidthSharded";
    using type = tt::tt_metal::address_generators::DefaultVirtualCoordWidthShardedAddressGenerator;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::HEIGHT_SHARDED, buffer_type, page_layout> {
    static constexpr char name[] = "HeightSharded";
    using type = tt::tt_metal::address_generators::DefaultVirtualCoordHeightShardedAddressGenerator;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::BLOCK_SHARDED, buffer_type, page_layout> {
    static constexpr char name[] = "BlockSharded";
    using type = tt::tt_metal::address_generators::DefaultVirtualCoordBlockShardedAddressGenerator;
};

constexpr bool is_sharded_tensor_layout(tt::tt_metal::TensorMemoryLayout tensor_layout) {
    return tensor_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
           tensor_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
           tensor_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
}

// reader code
template <typename T>
FORCE_INLINE constexpr Shape4D<T> build_wrapped_row_tensor_slice(T n_pages) {
    return Shape4D<T>{1, 1, 1, n_pages};
}

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
// #include "ttnn/tensor/types.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command_device.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_device.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "debug/dprint.h"
#include <type_traits>
#include <cstdint>

using ttnn::ccl::coord_t;
// For the future
using address_t = uint32_t;



using ttnn::ccl::Shape4D;
using tt::tt_metal::TensorMemoryLayout;

std::size_t get_flat_index_from_shape(const Shape4D<uint32_t> &shape, const Shape4D<uint32_t> &index) {
    std::size_t offset = index.x;
    std::size_t inner_volume = shape.x;
    offset += index.y * inner_volume;
    inner_volume *= shape.y;
    offset += index.z * inner_volume;
    inner_volume *= shape.z;
    offset += index.w * inner_volume;
    return offset;
}


namespace tt {
namespace tt_metal {
// TODO: Include directly from tt_metal or ttnn
enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };

// TODO: move from `buffer.hpp` into a commmon location
enum class BufferType {
    DRAM,
    L1,
    SYSTEM_MEMORY,
    L1_SMALL,
    TRACE,
};
}
}

using tt::tt_metal::BufferType;
using tt::tt_metal::Layout;

/// TODO: This is *mostly* duplicate (but updated and closer to the intended deisng) to
///       similar logic from worker_interleaved_ring_reduce_scatter_reader.cpp
///       -> BEFORE MERGE, DEPRECATE THAT ONE AND REPLACE WITH THIS ONE
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
    using type = tt::tt_metal::address_generators::DefaultWidthShardedAddressGenerator;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::HEIGHT_SHARDED, buffer_type, page_layout> {
    static constexpr char name[] = "HeightSharded";
    using type = tt::tt_metal::address_generators::DefaultHeightShardedAddressGenerator;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::BLOCK_SHARDED, buffer_type, page_layout> {
    static constexpr char name[] = "BlockSharded";
    using type = tt::tt_metal::address_generators::DefaultBlockShardedAddressGenerator;
};


constexpr bool is_sharded_tensor_layout(tt::tt_metal::TensorMemoryLayout tensor_layout) {
    return tensor_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
           tensor_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
           tensor_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
}

// reader code
template <typename T>
constexpr Shape4D<T> build_wrapped_row_tensor_slice(T n_pages) {
    return Shape4D<T>{1, 1, 1, n_pages};
}

//tt::tt_metal::Layout from ttnn/cpp/ttnn/tensor/types.hpp
template <tt::tt_metal::TensorMemoryLayout tensor_layout, tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
auto build_source_address_generator(std::size_t &arg_idx, address_t tensor_address, std::size_t page_size, uint32_t cb_id_in0) -> typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type {
    constexpr bool is_sharded = is_sharded_tensor_layout(tensor_layout);
    constexpr bool is_interleaved = tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    constexpr bool is_tile_page_layout = page_layout == tt::tt_metal::Layout::TILE;
    constexpr bool is_row_major_layout = page_layout == tt::tt_metal::Layout::ROW_MAJOR;
    static_assert(is_sharded || is_interleaved, "Only sharded and interleaved tensor layouts are supported but the unified address generator. A tensor layout not matching TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::BLOCK_SHARDED, or TensorMemoryLayout::INTERLEAVED was specified.");

    using addrgen_type = typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type;

    if constexpr (is_row_major_layout) {
        if constexpr (is_interleaved) {
            return addrgen_type{
                .bank_base_address = tensor_address, .page_size = page_size};
        } else if constexpr (is_sharded) {
            return tt::tt_metal::address_generators::build_sharded_addr_gen<tensor_layout>(
                tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(
                    0,//output_shard_grid_nrows,
                    0,//output_shard_grid_row_map,
                    0,//output_shard_grid_ncols,
                    0),//output_shard_grid_col_map),
                tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<tensor_layout>::type(
                    0,//output_tensor_shard_pages_per_shard_y,
                    0,//output_tensor_shard_pages_per_shard_x,
                    0,//output_tensor_shard_grid_height,
                    0,//output_tensor_shard_grid_width,
                    0,//output_tensor_shard_grid_start_y_logical,
                    0,//output_tensor_shard_grid_start_x_logical,
                    0//output_tensor_shard_grid_transposed
                ),
                page_size,
                tensor_address
            );
            ASSERT(false); // unimplemented and untested
        }
    } else if constexpr (is_tile_page_layout) {
        if constexpr (is_interleaved) {
            return addrgen_type{
                .bank_base_address = tensor_address, .page_size = page_size, .data_format = get_dataformat(cb_id_in0)};
        } else if constexpr (is_sharded) {
            ASSERT(false);//"Sharded support has not been added to ccl_send yet");
            return tt::tt_metal::address_generators::build_sharded_addr_gen<tensor_layout>(
                tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(
                    0,//output_shard_grid_nrows,
                    0,//output_shard_grid_row_map,
                    0,//output_shard_grid_ncols,
                    0),//output_shard_grid_col_map),
                tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<tensor_layout>::type(
                    0,//output_tensor_shard_pages_per_shard_y,
                    0,//output_tensor_shard_pages_per_shard_x,
                    0,//output_tensor_shard_grid_height,
                    0,//output_tensor_shard_grid_width,
                    0,//output_tensor_shard_grid_start_y_logical,
                    0,//output_tensor_shard_grid_start_x_logical,
                    0//output_tensor_shard_grid_transposed
                ),
                page_size,
                tensor_address
            );
        }
    }
}

/*
* CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time) dispatch
* implementations depending on those invocation parameters.
*/
void kernel_main() {
    std::size_t arg_idx = 0;
    using shape_t = Shape4D<uint32_t>;

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    constexpr TensorMemoryLayout tensor_layout = static_cast<TensorMemoryLayout>(get_compile_time_arg_val(0));
    constexpr BufferType buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
    constexpr Layout page_layout = static_cast<Layout>(get_compile_time_arg_val(2));
    constexpr ttnn::ccl::EriscDataMoverTerminationMode termination_mode = static_cast<ttnn::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(3));
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);

    // Load the input tensor spec
    address_t tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t num_commands = get_arg_val<address_t>(arg_idx++);

    // EDM Interface Parameters
    const ttnn::ccl::WorkerEdmInterfaceArgs edm_args = ttnn::ccl::build_from_args<ttnn::ccl::WorkerEdmInterfaceArgs>(arg_idx);
    // static_assert(tt_metal::is_compile_time_evaluated(edm_args.num_buffers_per_channel), "Number of buffers per channel was expected to resolve as compile time variable.");


    // Assuming whole page transmissions (which is the only mode we support at the moment)
    // -> however, wanted to call it out here to make it clear that we need to pull this
    //    out when we start enabling other modes
    const uint32_t packet_size_in_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_size = get_arg_val<uint32_t>(arg_idx++);
    auto tensor_addrgen = build_source_address_generator<tensor_layout, buffer_type, page_layout>(arg_idx, tensor_address, page_size, tt::CB::c_in0);
    volatile uint32_t* my_edm_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));

    // For now we only support single EDM connection
    ccl::edm::WorkerToEdmSender<termination_mode> sender(
        ttnn::ccl::WorkerXY(edm_args.edm_noc_x, edm_args.edm_noc_y),
        edm_args.edm_buffer_base_address,
        edm_args.num_buffers_per_channel,
        edm_args.edm_semaphore_address,
        packet_size_in_pages * page_size,
        my_edm_worker_semaphore_ptr);

    ttnn::ccl::cmd::CclCommandTensor command_tensor;

    for (std::size_t i = 0; i < num_commands; ++i) {
        // Generalized would be to get the command header info and then dispatch accordingly - if the command type is singular
        //
        // TODO: Turn this into a command iterator that initializes itself with the current arg_idx and then after that,
        //       the arg_idx never needs to be accessed again
        std::size_t old_arg_idx = arg_idx;
        ttnn::ccl::cmd::update_command_tensor(arg_idx, command_tensor);
        std::size_t new_arg_idx = arg_idx;

        {
            DPRINT << "cmd[" << (uint32_t)i << "]:\n";
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
            ASSERT(ccl_command.worker_pages_per_slice > 0);

            // CURRENTLY ONLY SUPPORTS WRAPPED TENSOR ITERATION COMMANDS
            // Implemented really inefficiently for now - in the future we can do more efficient packing and also change
            // the tensor read API to require the information in a more efficient way (less intermediate calculations)
            // const shape_t tensor_slice_start_offset = ttnn::ccl::build_from_args<shape_t>(arg_idx); // Should be RT
            shape_t valid_worker_slice_shape = build_wrapped_row_tensor_slice(command_tensor.worker_pages_per_slice); // Parametrizable by ct arg

            shape_t const& global_offset = command_tensor.tensor_slice_offset + command_tensor.worker_start_offset_in_slice;
            uint32_t curr_tile_id = get_flat_index_from_shape(command_tensor.tensor_shape, global_offset);

            uint32_t offset_into_worker_slice = 0;
            bool last_page_of_worker = false;
            for (uint32_t p = 0; p < command_tensor.worker_pages_per_slice; p += packet_size_in_pages) {
                uint32_t n_pages = std::min(packet_size_in_pages, command_tensor.worker_pages_per_slice - p);
                ASSERT(ccl_command.worker_start_offset_in_slice.w == 1);
                ASSERT(ccl_command.worker_start_offset_in_slice.z == 1);
                ASSERT(valid_worker_slice_shape.w == 1);
                ASSERT(valid_worker_slice_shape.z == 1);
                ASSERT(command_tensor.tensor_shape.w == 1);
                ASSERT(command_tensor.tensor_shape.z == 1);
                ASSERT(ccl_command.tensor_slice_shape.w == 1);
                ASSERT(ccl_command.tensor_slice_shape.z == 1);

                read_wrapped_chunk_from_output_tensor(
                    curr_tile_id,
                    offset_into_worker_slice,
                    ttnn::ccl::coord_t(command_tensor.worker_start_offset_in_slice.x, command_tensor.worker_start_offset_in_slice.y), // Offset into tensor slice
                    ttnn::ccl::coord_t(valid_worker_slice_shape.x, valid_worker_slice_shape.y),
                    // In tiles for tile layout
                    ttnn::ccl::coord_t(command_tensor.tensor_shape.x, command_tensor.tensor_shape.y),
                    ttnn::ccl::coord_t(command_tensor.tensor_slice_shape.x, command_tensor.tensor_slice_shape.y),
                    cb_id,
                    tensor_addrgen,
                    n_pages,
                    page_size,
                    last_page_of_worker);

                // Not optimal (doesn't overlap read/write) - but good for functional
                // bringup
                cb_wait_front(cb_id, n_pages);
                uint32_t l1_read_addr = get_read_ptr(cb_id);

                sender.wait_for_empty_write_slot();
                sender.send_payload_blocking(cb_id, n_pages, page_size);
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////

    sender.close();
}

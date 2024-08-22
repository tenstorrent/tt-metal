// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/tensor/types.hpp"
#include "debug/dprint.hpp"
#include <cstdint>

using ttnn::ccl::coord_t;
// For the future
using address_t = uint32_t;

namespace tt_metal {
template <typename T>
constexpr bool is_compile_time_evaluated(T value) {
    return std::is_constant_evaluated();
}
}




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
enum class Layout { ROW_MAJOR = 0, TILE = 1, INVALID = 2 };
}
}
/// TODO: This is *mostly* duplicate (but updated and closer to the intended deisng) to
///       similar logic from worker_interleaved_ring_reduce_scatter_reader.cpp
///       -> BEFORE MERGE, DEPRECATE THAT ONE AND REPLACE WITH THIS ONE
template <tt::tt_metal::TensorMemoryLayout tensor_layout, tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen {
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type, page_layout> {
    static constexpr bool is_dram = buffer_type == tt::tt_metal::BufferType::DRAM;
    using type = InterleavedAddrGen<is_dram>;
};
template <tt::tt_metal::BufferType buffer_type>
struct source_tensor_addrgen<tt::tt_metal::TensorMemoryLayout::INTERLEAVED, buffer_type, tt::tt_metal::Layout::TILE> {
    static constexpr bool is_dram = buffer_type == tt::tt_metal::BufferType::DRAM;
    using type = InterleavedAddrGenFast<is_dram>;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::WIDTH_SHARDED, buffer_type, page_layout> {
    using type = tt::tt_metal::address_generators::DefaultWidthShardedAddressGenerator;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::HEIGHT_SHARDED, buffer_type, page_layout> {
    using type = tt::tt_metal::address_generators::DefaultHeightShardedAddressGenerator;
};
template <tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
struct source_tensor_addrgen<TensorMemoryLayout::BLOCK_SHARDED, buffer_type, page_layout> {
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
    return {1, 1, 1, n_pages};
}

//tt::tt_metal::Layout from ttnn/cpp/ttnn/tensor/types.hpp
template <tt::tt_metal::TensorMemoryLayout tensor_layout, tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
auto build_source_address_generator(uint32_t &arg_idx, address_t tensor_address, std::size_t page_size) -> typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type {
    constexpr bool is_sharded = is_sharded_tensor_layout(tensor_layout);
    constexpr bool is_interleaved = tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    constexpr bool is_tile_page_layout = page_layout == tt::tt_metal::Layout::TILE;
    constexpr bool is_row_major_layout = page_layout == tt::tt_metal::Layout::ROW_MAJOR;
    static_assert(is_sharded || is_interleaved, "Only sharded and interleaved tensor layouts are supported but the unified address generator. A tensor layout not matching TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::BLOCK_SHARDED, or TensorMemoryLayout::INTERLEAVED was specified.");

    using addrgen_type = source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type;

    if constexpr (is_row_major_layout) {
        if constexpr (is_interleaved) {
        addrgen_type d = {
            .bank_base_address = tensor_address + output_start_addr_offset, .page_size = page_size};
        } else if constexpr (is_sharded) {
            auto d = tt::tt_metal::address_generators::build_sharded_addr_gen<output_tensor_memory_layout>(
                tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(output_shard_grid_nrows, output_shard_grid_row_map, output_shard_grid_ncols, output_shard_grid_col_map),
                tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<output_tensor_memory_layout>::type(
                    output_tensor_shard_pages_per_shard_y,
                    output_tensor_shard_pages_per_shard_x,
                    output_tensor_shard_grid_height,
                    output_tensor_shard_grid_width,
                    output_tensor_shard_grid_start_y_logical,
                    output_tensor_shard_grid_start_x_logical,
                    output_tensor_shard_grid_transposed
                ),
                page_size,
                dst_addr
            );
            ASSSERT(false); // unimplemented and untested
        }
    } else if constexpr (is_tile_page_layout) {
        if constexpr (is_interleaved) {
            addrgen_type d = {
                .bank_base_address = dst_addr, .page_size = tensor_address, .data_format = in0_df};
        } else if constexpr (is_sharded) {
        auto d = tt::tt_metal::address_generators::build_sharded_addr_gen<output_tensor_memory_layout>(
            tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(output_shard_grid_nrows, output_shard_grid_row_map, output_shard_grid_ncols, output_shard_grid_col_map),
            tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<output_tensor_memory_layout>::type(
                output_tensor_shard_pages_per_shard_y,
                output_tensor_shard_pages_per_shard_x,
                output_tensor_shard_grid_height,
                output_tensor_shard_grid_width,
                output_tensor_shard_grid_start_y_logical,
                output_tensor_shard_grid_start_x_logical,
                output_tensor_shard_grid_transposed
            ),
            page_size,
            dst_addr
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
    DPRINT << "CCL Send: Start\n";

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    constexpr TensorMemoryLayout tensor_layout = static_cast<tt::tt_metal::TensorMemoryLayout>(get_compile_time_arg_val(0));
    constexpr BufferType buffer_type = static_cast<tt::tt_metal::BufferType>(get_compile_time_arg_val(1));
    constexpr tt::tt_metal::Layout page_layout = static_cast<tt::tt_metal::BufferType>(get_compile_time_arg_val(2));
    constexpr ttnn::ccl::EriscDataMoverTerminationMode termination_mode = static_cast<ttnn::ccl::EriscDataMoverTerminationMode>(get_compile_time_arg_val(3));
    constexpr uint32_t cb_id = get_compile_time_arg_val(4);

    constexpr uint32_t next_raw_ct_arg_offset = 5;
    constexpr uint32_t edm_args_ct_args_offset = next_raw_ct_arg_offset;

    // Load the input tensor spec
    address_t tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t num_commands = get_arg_val<address_t>(arg_idx++);

    // Tensor Iterator Setup
    constexpr shape_t input_tensor_shape = ttnn::ccl::build_from_args<shape_t>(arg_idx); // Should be made full CT (common for all workers)
    constexpr shape_t tensor_window_shape = ttnn::ccl::build_from_args<shape_t>(arg_idx); // Should be made full CT (common for all workers)
    // constexpr shape_t tensor_slice_shape = build_from_args<shape_t>(arg_idx); // Should be made full CT (common for all workers)
    // Tensor iterator setup (custom to wrapped tensor iterator)


    // EDM Interface Parameters
    constexpr WorkerEdmInterfaceArgs edm_args = ttnn::ccl::build_from_args<WorkerEdmInterfaceArgs>(edm_args_ct_args_offset, arg_idx);
    constexpr uint32_t next_ct_arg = edm_args_ct_args_offset + ct_args_consumed<WorkerEdmInterfaceArgs>();
    static_assert(tt_metal::is_compile_time_evaluated(edm_args.num_buffers_per_channel), "Number of buffers per channel was expected to resolve as compile time variable.");


    // Assuming whole page transmissions (which is the only mode we support at the moment)
    // -> however, wanted to call it out here to make it clear that we need to pull this
    //    out when we start enabling other modes
    const uint32_t packet_size_in_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t page_size = get_arg_val<uint32_t>(arg_idx++);
    auto tensor_addrgen = build_source_address_generator<tensor_layout, buffer_type, page_layout>(arg_idx, tensor_address, page_size);
    volatile uint32_t* my_edm_worker_semaphore_ptr = reinterpret_cast<volatile uint32_t*>(get_semaphore(get_arg_val<uint32_t>(arg_idx++)));

    DPRINT << "CCL Send: Building sender\n";
    // For now we only support single EDM connection
    ccl::edm::WorkerToEdmSender<termination_mode> sender(
        ttnn::ccl::WorkerXY(edm_args.edm_noc_x, edm_args.edm_noc_y),
        edm_args.edm_buffer_base_address,
        edm_args.num_buffers_per_channel,
        edm_args.edm_semaphore_address,
        packet_size_in_pages * page_size,
        my_edm_worker_semaphore_ptr);

    DPRINT << "CCL Send: Running commands: " << num_commands << "\n";
    for (std::size_t i = 0; i < num_commands; ++i) {
        // Generalized would be to get the command header info and then dispatch accordingly - if the command type is singular
        //
        // TODO: Turn this into a command iterator that initializes itself with the current arg_idx and then after that,
        //       the arg_idx never needs to be accessed again
        auto ccl_command = ttnn::ccl::cmd::get_command(arg_idx);
        {
            DPRINT << "cmd[" << i << "]:\n";
            DPRINT << "\ttensor_slice_shape.w: " << ccl_command.tensor_slice_shape.w << "\n";
            DPRINT << "\ttensor_slice_shape.z: " << ccl_command.tensor_slice_shape.z << "\n";
            DPRINT << "\ttensor_slice_shape.y: " << ccl_command.tensor_slice_shape.y << "\n";
            DPRINT << "\ttensor_slice_shape.x: " << ccl_command.tensor_slice_shape.x << "\n";
            DPRINT << "\tensor_slice_offset.w: " << ccl_command.tensor_slice_offset.w << "\n";
            DPRINT << "\tensor_slice_offset.z: " << ccl_command.tensor_slice_offset.z << "\n";
            DPRINT << "\tensor_slice_offset.y: " << ccl_command.tensor_slice_offset.y << "\n";
            DPRINT << "\tensor_slice_offset.x: " << ccl_command.tensor_slice_offset.x << "\n";
            DPRINT << "\tworker_start_offset_in_slice.w: " << ccl_command.worker_start_offset_in_slice.w << "\n";
            DPRINT << "\tworker_start_offset_in_slice.z: " << ccl_command.worker_start_offset_in_slice.z << "\n";
            DPRINT << "\tworker_start_offset_in_slice.y: " << ccl_command.worker_start_offset_in_slice.y << "\n";
            DPRINT << "\tworker_start_offset_in_slice.x: " << ccl_command.worker_start_offset_in_slice.x << "\n";
            DPRINT << "\tworker_pages_per_slice.x: " << ccl_command.worker_pages_per_slice << "\n";

            // CURRENTLY ONLY SUPPORTS WRAPPED TENSOR ITERATION COMMANDS
            // Implemented really inefficiently for now - in the future we can do more efficient packing and also change
            // the tensor read API to require the information in a more efficient way (less intermediate calculations)
            const shape_t tensor_slice_start_offset = build_from_args<shape_t>(arg_idx); // Should be RT
            shape_t valid_worker_slice_shape = build_wrapped_row_tensor_slice(T n_pages); // Parametrizable by ct arg

            shape_t global_offset = tensor_slice_start_offset + ccl_command.worker_start_offset_in_slice;
            std::size_t curr_tile_id = get_flat_index_from_shape(input_tensor_shape, global_offset);

            std::size_t offset_into_worker_slice = 0;
            for (uint32_t p = 0; p < ccl_command.worker_pages_per_slice; p += packet_size_in_pages) {
                uint32_t n_pages = std::min(packet_size_in_pages, worker_slice_n_pages - p);
                read_wrapped_chunk_from_output_tensor(
                    curr_tile_id,
                    offset_into_worker_slice,
                    ccl_command.worker_start_offset_in_slice, // Offset into tensor slice
                    valid_worker_slice_shape,
                    // In tiles for tile layout
                    input_tensor_shape,
                    ccl_command.tensor_slice_shape,
                    cb_id,
                    tensor_addrgen,
                    n_pages,
                    args.page_size,
                    last_page_of_worker);

                // Not optimal (doesn't overlap read/write) - but good for functional
                // bringup
                sender.wait_for_empty_write_slot();
                sender.send_payload_blocking(cb_id_in0, num_pages_to_send, page_size);
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////

    sender.close();
}

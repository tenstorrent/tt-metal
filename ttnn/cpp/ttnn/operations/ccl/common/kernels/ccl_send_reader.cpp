// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command_device.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_device.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"
#include "debug/dprint.h"
#include "api/ttnn/tensor/enum_types.hpp"
#include "ttnn/operations/ccl/common/kernels/command_processor.hpp"
#include "ttnn/operations/ccl/common/kernels/ccl_send_utils.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"

#include <cstdint>

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr TensorMemoryLayout tensor_layout = static_cast<TensorMemoryLayout>(get_compile_time_arg_val(0));
constexpr BufferType buffer_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr Layout page_layout = static_cast<Layout>(get_compile_time_arg_val(2));
constexpr uint32_t cb_id = get_compile_time_arg_val(3);

#ifdef SHARDED_MEM_LAYOUT
static constexpr bool is_sharded_mode = true;
static constexpr uint32_t input_tensor_shard_grid_height = get_compile_time_arg_val(4);
static constexpr uint32_t input_tensor_shard_grid_width = get_compile_time_arg_val(5);
static constexpr uint32_t input_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(6);
static constexpr uint32_t input_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(7);
static constexpr uint32_t input_tensor_shard_pages_per_shard_y = get_compile_time_arg_val(8);
static constexpr uint32_t input_tensor_shard_pages_per_shard_x = get_compile_time_arg_val(9);
static constexpr bool input_tensor_shard_grid_transposed = get_compile_time_arg_val(10) != 0;
#else
static constexpr bool is_sharded_mode = false;
static constexpr uint32_t input_tensor_shard_grid_height = 0;
static constexpr uint32_t input_tensor_shard_grid_width = 0;
static constexpr uint32_t input_tensor_shard_grid_start_y_logical = 0;
static constexpr uint32_t input_tensor_shard_grid_start_x_logical = 0;
static constexpr uint32_t input_tensor_shard_pages_per_shard_y = 0;
static constexpr uint32_t input_tensor_shard_pages_per_shard_x = 0;
static constexpr bool input_tensor_shard_grid_transposed = false;
#endif

template <
    tt::tt_metal::TensorMemoryLayout tensor_layout,
    tt::tt_metal::BufferType buffer_type,
    tt::tt_metal::Layout page_layout>
auto build_source_address_generator(
    std::size_t& arg_idx, address_t tensor_address, std::size_t page_size, uint32_t cb_id_in0) ->
    typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type {
    constexpr bool is_sharded = is_sharded_tensor_layout(tensor_layout);
    constexpr bool is_interleaved = tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    constexpr bool is_tile_page_layout = page_layout == tt::tt_metal::Layout::TILE;
    constexpr bool is_row_major_layout = page_layout == tt::tt_metal::Layout::ROW_MAJOR;
    static_assert(
        is_sharded || is_interleaved,
        "Only sharded and interleaved tensor layouts are supported but the unified address generator. A tensor layout "
        "not matching TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::HEIGHT_SHARDED, "
        "TensorMemoryLayout::BLOCK_SHARDED, or TensorMemoryLayout::INTERLEAVED was specified.");

    using addrgen_type = typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type;

    if constexpr (tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        if constexpr (is_row_major_layout) {
            return addrgen_type{.bank_base_address = tensor_address, .page_size = page_size};
        } else {
            return addrgen_type{
                .bank_base_address = tensor_address, .page_size = page_size, .data_format = get_dataformat(cb_id_in0)};
        }
    } else if constexpr (
        tensor_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
        tensor_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
        tensor_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        size_t input_shard_grid_nrows = get_arg_val<uint32_t>(arg_idx++);
        const auto* const input_shard_grid_row_map = reinterpret_cast<const uint32_t* const>(get_arg_addr(arg_idx));
        arg_idx += input_shard_grid_nrows;
        size_t input_shard_grid_ncols = get_arg_val<uint32_t>(arg_idx++);
        const auto* const input_shard_grid_col_map = reinterpret_cast<const uint32_t* const>(get_arg_addr(arg_idx));
        arg_idx += input_shard_grid_ncols;

        return tt::tt_metal::address_generators::build_sharded_addr_gen<tensor_layout>(
            tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(
                input_shard_grid_nrows, input_shard_grid_row_map, input_shard_grid_ncols, input_shard_grid_col_map),
            typename tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<tensor_layout>::type(
                input_tensor_shard_pages_per_shard_y,
                input_tensor_shard_pages_per_shard_x,
                input_tensor_shard_grid_height,
                input_tensor_shard_grid_width,
                input_tensor_shard_grid_start_y_logical,
                input_tensor_shard_grid_start_x_logical,
                input_tensor_shard_grid_transposed),
            page_size,
            tensor_address);
    } else {
        ASSERT(false);
    }
}

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    std::size_t arg_idx = 0;

    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    // Load the input tensor spec
    address_t tensor_address = get_arg_val<address_t>(arg_idx++);
    address_t num_commands = get_arg_val<address_t>(arg_idx++);

    // Assuming whole page transmissions (which is the only mode we support at the moment)
    // -> however, wanted to call it out here to make it clear that we need to pull this
    //    out when we start enabling other modes
    const uint32_t packet_size_in_pages = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t payload_page_size = get_arg_val<uint32_t>(arg_idx++);
    auto tensor_addrgen = build_source_address_generator<tensor_layout, buffer_type, page_layout>(
        arg_idx, tensor_address, payload_page_size, tt::CB::c_in0);

    ttnn::ccl::cmd::CclCommandTensor command_tensor;

    // Don't use CBs because there appears to be a bug if we have the same producer/consumer core to a given CB
    // Instead, open up the CB and use it as a raw scratch space6

#ifdef DEBUG_PRINT_ENABLED
    DPRINT << "ccl_send_reader has " << (uint32_t)num_commands << " commands" << ENDL();
#endif

    for (std::size_t i = 0; i < num_commands; ++i) {
        // Generalized would be to get the command header info and then dispatch accordingly - if the command type is
        // singular
        //
        std::size_t old_arg_idx = arg_idx;
        ttnn::ccl::cmd::update_command_tensor(arg_idx, command_tensor);
        std::size_t new_arg_idx = arg_idx;

        {
            print_tensor_command(i, command_tensor);
            ASSERT(command_tensor.worker_pages_per_slice > 0);

            // CURRENTLY ONLY SUPPORTS WRAPPED TENSOR ITERATION COMMANDS
            // Implemented really inefficiently for now - in the future we can do more efficient packing and also change
            // the tensor read API to require the information in a more efficient way (less intermediate calculations)
            shape_t valid_worker_slice_shape =
                build_wrapped_row_tensor_slice(command_tensor.worker_pages_per_slice);  // Parametrizable by ct arg

            shape_t const& global_offset =
                command_tensor.tensor_slice_offset + command_tensor.worker_start_offset_in_slice;

            uint32_t curr_tile_id = get_flat_index_from_shape(command_tensor.tensor_shape, global_offset);

            uint32_t offset_into_worker_slice = 0;
            bool last_page_of_worker = false;
            for (uint32_t p = 0; p < command_tensor.worker_pages_per_slice; p += packet_size_in_pages) {
                cb_reserve_back(cb_id, packet_size_in_pages);
                const uint32_t local_l1_scratch_buffer_address =
                    get_write_ptr(cb_id) + sizeof(PACKET_HEADER_TYPE);

                uint32_t n_pages = std::min(packet_size_in_pages, command_tensor.worker_pages_per_slice - p);
                ASSERT(command_tensor.worker_start_offset_in_slice.w == 0);
                ASSERT(command_tensor.worker_start_offset_in_slice.z == 0);
                ASSERT(valid_worker_slice_shape.w == 1);
                ASSERT(valid_worker_slice_shape.z == 1);
                ASSERT(command_tensor.tensor_shape.w == 1);
                ASSERT(command_tensor.tensor_shape.z == 1);
                ASSERT(command_tensor.tensor_slice_shape.w == 1);
                ASSERT(command_tensor.tensor_slice_shape.z == 1);

                DPRINT << "iter " << p << " curr_tile_id: " << curr_tile_id << ENDL();
                DPRINT << "local_l1_scratch_buffer_address: " << local_l1_scratch_buffer_address << ENDL();

                read_wrapped_chunk_from_output_tensor_to_address(
                    curr_tile_id,
                    offset_into_worker_slice,
                    ttnn::ccl::coord_t(
                        command_tensor.worker_start_offset_in_slice.x,
                        command_tensor.worker_start_offset_in_slice.y),  // Offset into tensor slice
                    ttnn::ccl::coord_t(valid_worker_slice_shape.x, valid_worker_slice_shape.y),
                    // In tiles for tile layout
                    ttnn::ccl::coord_t(command_tensor.tensor_shape.x, command_tensor.tensor_shape.y),
                    ttnn::ccl::coord_t(command_tensor.tensor_slice_shape.x, command_tensor.tensor_slice_shape.y),
                    local_l1_scratch_buffer_address,
                    tensor_addrgen,
                    n_pages,
                    payload_page_size,
                    last_page_of_worker);

                cb_push_back(cb_id, packet_size_in_pages);
            }
        }
    }
    ////////////////////////////////////////////////////////////////////////////////////

    DPRINT << "ccl_send_reader done\n";
}

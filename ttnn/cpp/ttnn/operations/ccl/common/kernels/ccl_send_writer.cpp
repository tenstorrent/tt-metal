// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/common/uops/ccl_command_device.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_device.hpp"
#include "ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"
#include "api/ttnn/tensor/enum_types.hpp"
#include "ttnn/operations/ccl/common/kernels/command_processor.hpp"
#include "ttnn/operations/ccl/common/kernels/ccl_send_utils.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

#include "debug/dprint.h"
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
    address_t dest_address = get_arg_val<address_t>(arg_idx++);
    address_t num_commands = get_arg_val<address_t>(arg_idx++);

    // Assuming whole page transmissions (which is the only mode we support at the moment)
    // -> however, wanted to call it out here to make it clear that we need to pull this
    //    out when we start enabling other modes
    const size_t packet_size_in_pages = get_arg_val<uint32_t>(arg_idx++);
    const size_t payload_page_size = get_arg_val<uint32_t>(arg_idx++);
    const size_t l1_scratch_page_size = payload_page_size + sizeof(PACKET_HEADER_TYPE);
    const size_t forward_direction_num_hops = get_arg_val<uint32_t>(arg_idx++);
    const size_t backward_direction_num_hops = get_arg_val<uint32_t>(arg_idx++);
    const bool has_forward_fabric_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    auto forward_fabric_sender =
        has_forward_fabric_connection
            ? tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx)
            : tt::tt_fabric::WorkerToFabricEdmSender();
    const bool has_backward_fabric_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    auto backward_fabric_sender =
        has_backward_fabric_connection
            ? tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx)
            : tt::tt_fabric::WorkerToFabricEdmSender();

    constexpr size_t num_args_per_sync_signal_sender = 3;
    const bool must_send_sync_signals = get_arg_val<uint32_t>(arg_idx++) != 0;
    auto num_sync_signals = must_send_sync_signals ? get_arg_val<uint32_t>(arg_idx++) : 0;
    auto sync_details_arg_idx = arg_idx;
    arg_idx += num_sync_signals * num_args_per_sync_signal_sender;

    auto tensor_addrgen = build_source_address_generator<tensor_layout, buffer_type, page_layout>(
        arg_idx, dest_address, payload_page_size, tt::CB::c_in0);

    if (has_forward_fabric_connection) {
        DPRINT << "Opening forward fabric connection\n";
        forward_fabric_sender.open();
        DPRINT << "Forward fabric connection opened\n";
    }
    if (has_backward_fabric_connection) {
        DPRINT << "Opening backward fabric connection\n";
        backward_fabric_sender.open();
        DPRINT << "Backward fabric connection opened\n";
    }

    ttnn::ccl::cmd::CclCommandTensor command_tensor;

#ifdef DEBUG_PRINT_ENABLED
    DPRINT << "ccl_send_writer has " << (uint32_t)num_commands << " commands" << ENDL();
#endif

    size_t some_buffering_addr = 0;

    for (std::size_t i = 0; i < num_commands; ++i) {
        // Generalized would be to get the command header info and then dispatch accordingly - if the command type is
        // singular
        //
        std::size_t old_arg_idx = arg_idx;
        ttnn::ccl::cmd::update_command_tensor(arg_idx, command_tensor);
        std::size_t new_arg_idx = arg_idx;

        {
            // print_tensor_command(i, command_tensor);
            ASSERT(command_tensor.worker_pages_per_slice > 0);

            // CURRENTLY ONLY SUPPORTS WRAPPED TENSOR ITERATION COMMANDS
            // Implemented really inefficiently for now - in the future we can do more efficient packing and also change
            // the tensor read API to require the information in a more efficient way (less intermediate calculations)
            shape_t valid_worker_slice_shape =
                build_wrapped_row_tensor_slice(command_tensor.worker_pages_per_slice);  // Parametrizable by ct arg

            shape_t const& global_offset =
                command_tensor.tensor_slice_offset + command_tensor.worker_start_offset_in_slice;

            uint32_t curr_page_idx = get_flat_index_from_shape(command_tensor.tensor_shape, global_offset);

            uint32_t offset_into_worker_slice = 0;
            DPRINT << "Outside loop\n";
            DPRINT << "worker_pages_per_slice: " << command_tensor.worker_pages_per_slice << ENDL();
            DPRINT << "payload_page_size: " << (uint32_t)payload_page_size << ENDL();
            // DPRINT << "packet_size_in_pages: " << packet_size_in_pages << ENDL();
            for (uint32_t p = 0; p < command_tensor.worker_pages_per_slice; p += packet_size_in_pages) {
                DPRINT << "Packet loop\n";
                uint32_t n_pages = std::min<uint32_t>(packet_size_in_pages, command_tensor.worker_pages_per_slice - p);

                ASSERT(command_tensor.worker_start_offset_in_slice.w == 0);
                ASSERT(command_tensor.worker_start_offset_in_slice.z == 0);
                ASSERT(valid_worker_slice_shape.w == 1);
                ASSERT(valid_worker_slice_shape.z == 1);
                ASSERT(command_tensor.tensor_shape.w == 1);
                ASSERT(command_tensor.tensor_shape.z == 1);
                ASSERT(command_tensor.tensor_slice_shape.w == 1);
                ASSERT(command_tensor.tensor_slice_shape.z == 1);

                DPRINT << "iter " << p << " curr_tile_id: " << curr_page_idx << ENDL();

                DPRINT << "cb_wait_front\n";
                cb_wait_front(cb_id, n_pages);
                DPRINT << "cb_wait_front done\n";
                size_t l1_read_addr = get_read_ptr(cb_id);
                some_buffering_addr = l1_read_addr;

                mcast_payload_chunk_to_output_tensor_address<tensor_layout, page_layout>(
                    curr_page_idx,
                    offset_into_worker_slice,
                    command_tensor.worker_start_offset_in_slice,  // worker_slice_offset
                    valid_worker_slice_shape,                     // worker_slice_shape
                    command_tensor,
                    l1_read_addr,
                    n_pages,
                    payload_page_size,
                    l1_scratch_page_size,
                    has_forward_fabric_connection,
                    has_backward_fabric_connection,
                    forward_fabric_sender,
                    backward_fabric_sender,
                    forward_direction_num_hops,
                    backward_direction_num_hops,
                    tensor_addrgen);

                DPRINT << "cb_pop_front\n";
                cb_pop_front(cb_id, n_pages);
                DPRINT << "cb_pop_front done\n";
            }
            DPRINT << "Packet loop done\n";
        }
        DPRINT << "Outside loop done\n";
    }
    DPRINT << "ccl_send_writer done main loop - enterring teardown\n";

    if (must_send_sync_signals) {
        DPRINT << "ccl_send_writer Sending payload completion sync signals\n";
        ASSERT(some_buffering_addr != 0);
        some_buffering_addr =
            (some_buffering_addr + (sizeof(PACKET_HEADER_TYPE))) & ~(sizeof(PACKET_HEADER_TYPE) - 1);

        mcast_sync_signal_to_addr(
            some_buffering_addr,
            sync_details_arg_idx,
            has_forward_fabric_connection,
            has_backward_fabric_connection,
            forward_fabric_sender,
            backward_fabric_sender,
            forward_direction_num_hops,
            backward_direction_num_hops,
            num_sync_signals);
    }

    DPRINT << "ccl_send_writer closing connections\n";
    if (has_forward_fabric_connection) {
        forward_fabric_sender.close();
    }
    if (has_backward_fabric_connection) {
        backward_fabric_sender.close();
    }
    ////////////////////////////////////////////////////////////////////////////////////
    DPRINT << "ccl_send_writer done\n";
}

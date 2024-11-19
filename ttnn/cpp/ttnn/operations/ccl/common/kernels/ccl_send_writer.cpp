// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types.hpp"
#include "tt_metal/impl/buffers/buffer_constants.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/uops/ccl_command_device.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/types/ccl_types_device.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/worker_edm_adapters.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"
#include "debug/dprint.h"
#include "ttnn/cpp/ttnn/tensor/enum_types.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/command_processor.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_transmission.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"

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
static constexpr uint32_t input_tensor_shard_grid_height = get_compile_time_arg_val(5);
static constexpr uint32_t input_tensor_shard_grid_width = get_compile_time_arg_val(6);
static constexpr uint32_t input_tensor_shard_grid_start_y_logical = get_compile_time_arg_val(7);
static constexpr uint32_t input_tensor_shard_grid_start_x_logical = get_compile_time_arg_val(8);
static constexpr uint32_t input_tensor_shard_pages_per_shard_y = get_compile_time_arg_val(9);
static constexpr uint32_t input_tensor_shard_pages_per_shard_x = get_compile_time_arg_val(10);
static constexpr bool input_tensor_shard_grid_transposed = get_compile_time_arg_val(11) != 0;
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


template <tt::tt_metal::TensorMemoryLayout tensor_layout, tt::tt_metal::BufferType buffer_type, tt::tt_metal::Layout page_layout>
auto build_source_address_generator(std::size_t &arg_idx, address_t tensor_address, std::size_t page_size, uint32_t cb_id_in0) -> typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type {
    constexpr bool is_sharded = is_sharded_tensor_layout(tensor_layout);
    constexpr bool is_interleaved = tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED;
    constexpr bool is_tile_page_layout = page_layout == tt::tt_metal::Layout::TILE;
    constexpr bool is_row_major_layout = page_layout == tt::tt_metal::Layout::ROW_MAJOR;
    static_assert(is_sharded || is_interleaved, "Only sharded and interleaved tensor layouts are supported but the unified address generator. A tensor layout not matching TensorMemoryLayout::WIDTH_SHARDED, TensorMemoryLayout::HEIGHT_SHARDED, TensorMemoryLayout::BLOCK_SHARDED, or TensorMemoryLayout::INTERLEAVED was specified.");

    using addrgen_type = typename source_tensor_addrgen<tensor_layout, buffer_type, page_layout>::type;

    if constexpr (tensor_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        if constexpr (is_row_major_layout) {
            return addrgen_type{
                .bank_base_address = tensor_address, .page_size = page_size};
        } else {
            return addrgen_type{
                .bank_base_address = tensor_address, .page_size = page_size, .data_format = get_dataformat(cb_id_in0)};
        }
    } else if constexpr (
        tensor_layout == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED ||
        tensor_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
        tensor_layout == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED) {
        size_t input_shard_grid_nrows = get_arg_val<uint32_t>(arg_idx++);
        const auto * const input_shard_grid_row_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
        arg_idx += input_shard_grid_nrows;
        size_t input_shard_grid_ncols = get_arg_val<uint32_t>(arg_idx++);
        const auto * const input_shard_grid_col_map = reinterpret_cast<const uint32_t * const>(get_arg_addr(arg_idx));
        arg_idx += input_shard_grid_ncols;

        return tt::tt_metal::address_generators::build_sharded_addr_gen<tensor_layout>(
            tt::tt_metal::address_generators::HarvestedWormholeWorkerToNocLookup(
                input_shard_grid_nrows,
                input_shard_grid_row_map,
                input_shard_grid_ncols,
                input_shard_grid_col_map),
            typename tt::tt_metal::address_generators::DeviceShardSpecTypeGetter<tensor_layout>::type(
                input_tensor_shard_pages_per_shard_y,
                input_tensor_shard_pages_per_shard_x,
                input_tensor_shard_grid_height,
                input_tensor_shard_grid_width,
                input_tensor_shard_grid_start_y_logical,
                input_tensor_shard_grid_start_x_logical,
                input_tensor_shard_grid_transposed
            ),
            page_size,
            tensor_address
        );
    } else {
        ASSERT(false);
    }
}

std::pair<WorkerXY, uint32_t> get_noc_address_components(uint64_t noc_addr) {
    const size_t bank_addr = noc_addr & 0xFFFFFFFF;
    const size_t noc_x = (noc_addr >> NOC_ADDR_LOCAL_BITS) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1);
    const size_t noc_y = (noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1);
    return {WorkerXY(noc_x, noc_y), bank_addr};
}

/*
* CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time) dispatch
* implementations depending on those invocation parameters.
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
    const size_t page_size = get_arg_val<uint32_t>(arg_idx++);
    const size_t forward_direction_num_hops = get_arg_val<uint32_t>(arg_idx++);
    const size_t backward_direction_num_hops = get_arg_val<uint32_t>(arg_idx++);
    const bool has_forward_fabric_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    auto forward_fabric_sender = has_forward_fabric_connection ? tt::fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx) : tt::fabric::WorkerToFabricEdmSender();
    const bool has_backward_fabric_connection = get_arg_val<uint32_t>(arg_idx++) != 0;
    auto backward_fabric_sender = has_backward_fabric_connection ? tt::fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg_idx) : tt::fabric::WorkerToFabricEdmSender();

    constexpr size_t num_args_per_sync_signal_sender = 3;
    const bool must_send_sync_signals = get_arg_val<uint32_t>(arg_idx++) != 0;
    auto num_sync_signals = must_send_sync_signals ? get_arg_val<uint32_t>(arg_idx++) : 0;
    auto sync_details_arg_idx = arg_idx;
    arg_idx += num_sync_signals * num_args_per_sync_signal_sender;

    auto tensor_addrgen = build_source_address_generator<tensor_layout, buffer_type, page_layout>(arg_idx, dest_address, page_size, tt::CB::c_in0);

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
        // Generalized would be to get the command header info and then dispatch accordingly - if the command type is singular
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
            // const shape_t tensor_slice_start_offset = ttnn::ccl::build_from_args<shape_t>(arg_idx); // Should be RT
            shape_t valid_worker_slice_shape = build_wrapped_row_tensor_slice(command_tensor.worker_pages_per_slice); // Parametrizable by ct arg

            shape_t const& worker_start_offset_global = worker_wrapped_offset_to_coord(command_tensor.tensor_slice_shape, command_tensor.worker_start_offset_in_slice);
            shape_t const& global_offset = command_tensor.tensor_slice_offset + worker_start_offset_global;

            uint32_t curr_page_idx = get_flat_index_from_shape(command_tensor.tensor_shape, global_offset);

            // DPRINT << "valid_worker_slice_shape.w: " << valid_worker_slice_shape.w << ENDL();
            // DPRINT << "valid_worker_slice_shape.z: " << valid_worker_slice_shape.z << ENDL();
            // DPRINT << "valid_worker_slice_shape.y: " << valid_worker_slice_shape.y << ENDL();
            // DPRINT << "valid_worker_slice_shape.x: " << valid_worker_slice_shape.x << ENDL();
            // DPRINT << "global_offset.w: " << global_offset.w << ENDL();
            // DPRINT << "global_offset.z: " << global_offset.z << ENDL();
            // DPRINT << "global_offset.y: " << global_offset.y << ENDL();
            // DPRINT << "global_offset.x: " << global_offset.x << ENDL();
            // DPRINT << "curr_page_idx: " << curr_page_idx << ENDL();

            uint32_t offset_into_worker_slice = 0;
            bool last_page_of_worker = false;
            DPRINT << "Outside loop\n";
            DPRINT << "worker_pages_per_slice: " << command_tensor.worker_pages_per_slice << ENDL();
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

                DPRINT << "cb_wait_front\n";
                cb_wait_front(cb_id, n_pages);
                DPRINT << "cb_wait_front done\n";
                size_t l1_read_addr = get_read_ptr(cb_id);
                some_buffering_addr = l1_read_addr;

                auto const worker_slice_offset = command_tensor.worker_start_offset_in_slice;
                auto const worker_slice_shape = valid_worker_slice_shape;

                size_t contig_pages_advanced = 1;
                for (size_t i = 0; i < n_pages; i += contig_pages_advanced) {
                    // DPRINT << "Contig loop\n";
                    auto const [noc_addr, contig_pages] = get_noc_addr_and_contiguous_pages<tensor_layout, page_layout>(
                        curr_page_idx,
                        offset_into_worker_slice,
                        worker_slice_offset,
                        tensor_addrgen,
                        command_tensor.tensor_slice_shape);

                    // DPRINT << "CHKPT\n";
                    contig_pages_advanced = std::min<size_t>(contig_pages, n_pages);

                    // Perform the local write and also the mcast packet writes
                    // TODO: make all this driven by the command attributes that are passed to the
                    //       kernel so it doesn't end up being all-gather/mcast specific
                    {  /// ALL JUST FABRIC MCAST LOGIC - PULL INTO SEPARATE FUNCTIon
                        // Initiate the local write
                        const size_t payload_l1_address = l1_read_addr + sizeof(tt::fabric::PacketHeader);
                        size_t payload_size_bytes = contig_pages_advanced * page_size;
                        // local chip write
                        noc_async_write(payload_l1_address, noc_addr, payload_size_bytes);
                        // DPRINT << "CHKPT2\n";
                        // Write the mcast packet (forward)
                        const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc_addr);
                        // DPRINT << "CHKPT3\n";
                        size_t packet_send_size_bytes = payload_size_bytes + sizeof(tt::fabric::PacketHeader);
                        if (has_forward_fabric_connection) {
                            // DPRINT << "Forward fabric connection\n";
                            static_assert(((sizeof(tt::fabric::PacketHeader) - 1) & sizeof(tt::fabric::PacketHeader)) == 0, "sizeof(sizeof(tt::fabric::PacketHeader)) is not a power of two which violates the below assertion");
                            // ASSERT(l1_read_addr & (sizeof(tt::fabric::PacketHeader) - 1) == 0);
                            auto &pkt_hdr = *reinterpret_cast<tt::fabric::PacketHeader*>(l1_read_addr);
                            pkt_hdr.to_write()
                                .to_chip_multicast(tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(forward_direction_num_hops/* - 1*/)})
                                .to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
                                    dest_addr, packet_send_size_bytes, static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y)
                                });
                            forward_fabric_sender.wait_for_empty_write_slot();
                            forward_fabric_sender.send_payload_flush_blocking_from_address(l1_read_addr, packet_send_size_bytes);
                            // DPRINT << "FWD_PKT @ " << (uint64_t)l1_read_addr << "\n";
                            // DPRINT << "FWD_PKT " << (uint64_t)*reinterpret_cast<volatile uint64_t*>(l1_read_addr) << "\n";
                        }

                        // Write the mcast packet (backward)
                        if (has_backward_fabric_connection) {
                            // DPRINT << "Backward fabric connection\n";
                            auto &pkt_hdr = *reinterpret_cast<tt::fabric::PacketHeader*>(l1_read_addr);
                            pkt_hdr.to_write()
                                .to_chip_multicast(tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(backward_direction_num_hops/* - 1*/)})
                                .to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
                                    dest_addr, packet_send_size_bytes, static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y)
                                });
                            // DPRINT << "Backward waiting for empty write slot\n";
                            backward_fabric_sender.wait_for_empty_write_slot();
                            // DPRINT << "Backward sending payload\n";
                            backward_fabric_sender.send_payload_non_blocking_from_address(l1_read_addr, packet_send_size_bytes);
                            // Commit to memory
                            // DPRINT << "BWD_PKT @ " << (uint64_t)l1_read_addr << "\n";
                            // DPRINT << "BWD_PKT " << (uint64_t)*reinterpret_cast<volatile uint64_t*>(l1_read_addr) << "\n";
                        }
                    }

                    // DPRINT << "advance_worker_global_page\n";
                    ttnn::ccl::v2::advance_worker_global_page(
                        curr_page_idx, // Updated internally
                        offset_into_worker_slice,
                        worker_slice_offset,
                        worker_slice_shape,
                        command_tensor.tensor_slice_shape,
                        command_tensor.tensor_shape,
                        contig_pages,
                        last_page_of_worker
                    );


                    // // build headers and write to the output cb
                    // DPRINT << "noc_async_write_barrier\n";
                    noc_async_write_barrier(); // since our last call to
                    DPRINT << "cb_pop_front\n";
                    cb_pop_front(cb_id, packet_size_in_pages);
                    DPRINT << "cb_pop_front done\n";

                    l1_read_addr += contig_pages_advanced * page_size;
                }

            }
            DPRINT << "Packet loop done\n";
        }
        DPRINT << "Outside loop done\n";
    }
    DPRINT << "ccl_send_writer done main loop - enterring teardown\n";

    if (must_send_sync_signals) {
        DPRINT << "ccl_send_writer Sending payload completion sync signals\n";
        ASSERT(some_buffering_addr != 0);
        some_buffering_addr = (some_buffering_addr + (sizeof(tt::fabric::PacketHeader))) & ~(sizeof(tt::fabric::PacketHeader) - 1);
        auto send_sync_signal = [](
            size_t pkt_addr,
            tt::fabric::WorkerToFabricEdmSender &fabric_connection,
            size_t remote_sem_noc_x, size_t remote_sem_noc_y,
            size_t remote_sem_l1_addr,
            size_t directional_num_hops) {
            static_assert(((sizeof(tt::fabric::PacketHeader) - 1) & sizeof(tt::fabric::PacketHeader)) == 0, "sizeof(sizeof(tt::fabric::PacketHeader)) is not a power of two which violates the below assertion");
            ASSERT((pkt_addr & (sizeof(tt::fabric::PacketHeader) - 1)) == 0);

            auto &pkt_hdr = *reinterpret_cast<tt::fabric::PacketHeader*>(pkt_addr);
            pkt_hdr.to_atomic_inc()
                .to_chip_multicast(tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(directional_num_hops/* - 1*/)})
                .to_noc_unicast_atomic_inc(tt::fabric::NocUnicastAtomicIncCommandHeader{
                    remote_sem_l1_addr, 1, 32, static_cast<uint8_t>(remote_sem_noc_x), static_cast<uint8_t>(remote_sem_noc_y)
                });
            print_pkt_header(&pkt_hdr);
            fabric_connection.wait_for_empty_write_slot();
            fabric_connection.send_payload_flush_blocking_from_address(pkt_addr, pkt_hdr.get_payload_size_including_header());
        };

        for (size_t i = 0; i < num_sync_signals; ++i) {
            auto dest_sem_addr = get_semaphore(get_arg_val<uint32_t>(sync_details_arg_idx++));
            auto dest_noc_x = get_arg_val<uint32_t>(sync_details_arg_idx++);
            auto dest_noc_y = get_arg_val<uint32_t>(sync_details_arg_idx++);

            auto sem_inc_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, dest_sem_addr);
            noc_semaphore_inc(sem_inc_noc_addr, 1);
            DPRINT << "Send sync signal to " << (uint64_t)sem_inc_noc_addr << "\n";
            DPRINT << "dest_sem_addr: " << (uint32_t)dest_sem_addr << "\n";
            DPRINT << "dest_noc_x: " << (uint32_t)dest_noc_x << "\n";
            DPRINT << "dest_noc_y: " << (uint32_t)dest_noc_y << "\n";
            DPRINT << "forward_direction_num_hops: " << (uint32_t)forward_direction_num_hops << "\n";
            DPRINT << "backward_direction_num_hops: " << (uint32_t)backward_direction_num_hops << "\n";

            if (has_forward_fabric_connection) {
                const size_t pkt_addr = some_buffering_addr;
                send_sync_signal(
                    pkt_addr,
                    forward_fabric_sender,
                    dest_noc_x,
                    dest_noc_y,
                    dest_sem_addr,
                    forward_direction_num_hops);
            }
            if (has_backward_fabric_connection) {
                const size_t pkt_addr = some_buffering_addr;
                send_sync_signal(
                    pkt_addr,
                    backward_fabric_sender,
                    dest_noc_x,
                    dest_noc_y,
                    dest_sem_addr,
                    backward_direction_num_hops);
            }
        }
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

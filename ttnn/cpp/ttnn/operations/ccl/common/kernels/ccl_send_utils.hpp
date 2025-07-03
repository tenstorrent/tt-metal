// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_edm_packet_transmission.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"

#include "debug/dprint.h"
#include <cstdint>

//------------------------------------------------------------------------------
// Section 1: Generic Utility Functions
//------------------------------------------------------------------------------

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename AddrGen>
std::pair<uint64_t, size_t> legacy_get_noc_addr_and_contiguous_pages(
    uint32_t curr_page_idx,
    const uint32_t offset_into_worker_slice,
    const ttnn::ccl::Shape4D<uint32_t>& offset_worker_slice,
    const AddrGen& address_generator,
    const ttnn::ccl::Shape4D<uint32_t>& tensor_slice_shape,
    uint8_t noc_id = noc_index) {
    if constexpr (TENSOR_LAYOUT == tt::tt_metal::TensorMemoryLayout::INTERLEAVED) {
        static constexpr uint32_t offset = 0;
        uint64_t dst_noc_addr = get_noc_addr(curr_page_idx, address_generator, offset, noc_id);
        return {dst_noc_addr, 1};
    } else {
        static_assert(
            TENSOR_LAYOUT == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED ||
            TENSOR_LAYOUT == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED ||
            TENSOR_LAYOUT == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED);
        if constexpr (MEM_LAYOUT == tt::tt_metal::Layout::ROW_MAJOR) {
            ASSERT(false);  // unimplemented
            return {0, 0};
        } else {
            static_assert(MEM_LAYOUT == tt::tt_metal::Layout::TILE);
            // TODO: Make d.get_noc_addr work on host + device
            auto const& [noc_yx, page_offset, contig_pages_] =
                address_generator.get_page_location_with_contiguous_pages_in_row_in_bank(curr_page_idx);
            /*
             * Shared with `read_wrapped_chunk_from_output_tensor`
             */
            uint32_t flattened_offset_worker_slice =
                ttnn::ccl::v2::flattened_index(tensor_slice_shape, offset_worker_slice);
            uint32_t contig_until_edge_of_tensor_slice =
                tensor_slice_shape.x -
                ((flattened_offset_worker_slice + offset_into_worker_slice) % tensor_slice_shape.x);

            size_t contig_pages = std::min<int32_t>(contig_pages_, contig_until_edge_of_tensor_slice);
            uint64_t dst_noc_addr = get_noc_addr(
                static_cast<uint32_t>(noc_yx.noc_x),
                noc_yx.noc_y,
                address_generator.bank_base_address + (page_offset * address_generator.page_size) + 0,
                noc_id);
            return {dst_noc_addr, contig_pages};
        }
    }
}

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename AddrGen>
FORCE_INLINE std::pair<uint64_t, size_t> legacy_get_noc_addr_and_contiguous_pages_for_fabric_write(
    uint32_t curr_page_idx,
    const uint32_t offset_into_worker_slice,
    const ttnn::ccl::Shape4D<uint32_t>& offset_worker_slice,
    const AddrGen& address_generator,
    const ttnn::ccl::Shape4D<uint32_t>& tensor_slice_shape) {
    return legacy_get_noc_addr_and_contiguous_pages<TENSOR_LAYOUT, MEM_LAYOUT, AddrGen>(
        curr_page_idx, offset_into_worker_slice, offset_worker_slice, address_generator, tensor_slice_shape, 0);
}

std::pair<WorkerXY, uint32_t> get_noc_address_components(uint64_t noc_addr) {
    const size_t bank_addr = noc_addr & 0xFFFFFFFF;
    const size_t noc_x = (noc_addr >> NOC_ADDR_LOCAL_BITS) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1);
    const size_t noc_y =
        (noc_addr >> (NOC_ADDR_LOCAL_BITS + NOC_ADDR_NODE_ID_BITS)) & ((1 << NOC_ADDR_NODE_ID_BITS) - 1);
    return {WorkerXY(noc_x, noc_y), bank_addr};
}

//------------------------------------------------------------------------------
// Section 2: Multicast Write with Fabric Functions
//------------------------------------------------------------------------------

void mcast_contig_pages_to_noc_address(
    uint64_t noc0_dest_addr,
    size_t l1_read_addr,
    size_t contig_pages_advanced,
    size_t payload_page_size,
    bool has_forward_fabric_connection,
    bool has_backward_fabric_connection,
    tt::tt_fabric::WorkerToFabricEdmSender& forward_fabric_sender,
    tt::tt_fabric::WorkerToFabricEdmSender& backward_fabric_sender,
    size_t forward_direction_num_hops,
    size_t backward_direction_num_hops) {
    const size_t payload_size_bytes = contig_pages_advanced * payload_page_size;
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_addr);
    const size_t payload_l1_address = l1_read_addr + sizeof(PACKET_HEADER_TYPE);

    // Local chip write
    noc_async_write(
        payload_l1_address,
        // We are writing out from local core so we need to normalize to our noc
        // if the target is a virtual coord this is actually redundant but for DRAM
        // coords it is necessary
        get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr, noc_index),
        payload_size_bytes);
    size_t packet_send_size_bytes = payload_size_bytes + sizeof(PACKET_HEADER_TYPE);

    // Forward fabric connection
    if (has_forward_fabric_connection) {
        static_assert(
            is_power_of_2(sizeof(PACKET_HEADER_TYPE)),
            "sizeof(tt::tt_fabric::PacketHeader) is not a power of two which violates the below assertion");

        auto& pkt_hdr = *reinterpret_cast<PACKET_HEADER_TYPE*>(l1_read_addr);
        pkt_hdr
            .to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(forward_direction_num_hops)})
            .to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_addr}, packet_send_size_bytes);
        forward_fabric_sender.wait_for_empty_write_slot();
        forward_fabric_sender.send_payload_flush_blocking_from_address(l1_read_addr, packet_send_size_bytes);
    }

    // Backward fabric connection
    if (has_backward_fabric_connection) {
        auto& pkt_hdr = *reinterpret_cast<PACKET_HEADER_TYPE*>(l1_read_addr);
        pkt_hdr
            .to_chip_multicast(
                tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(backward_direction_num_hops)})
            .to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_addr}, packet_send_size_bytes);
        backward_fabric_sender.wait_for_empty_write_slot();
        backward_fabric_sender.send_payload_non_blocking_from_address(l1_read_addr, packet_send_size_bytes);
    }
}

template <tt::tt_metal::TensorMemoryLayout TENSOR_LAYOUT, tt::tt_metal::Layout MEM_LAYOUT, typename AddrGen>
void mcast_payload_chunk_to_output_tensor_address(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
    const shape_t& worker_slice_offset,
    const shape_t& worker_slice_shape,
    const ttnn::ccl::cmd::CclCommandTensor& command_tensor,
    size_t l1_read_addr,
    size_t n_pages,
    size_t payload_page_size,
    size_t l1_scratch_page_size,
    bool has_forward_fabric_connection,
    bool has_backward_fabric_connection,
    tt::tt_fabric::WorkerToFabricEdmSender& forward_fabric_sender,
    tt::tt_fabric::WorkerToFabricEdmSender& backward_fabric_sender,
    size_t forward_direction_num_hops,
    size_t backward_direction_num_hops,
    const AddrGen& tensor_addrgen) {
    size_t contig_pages_advanced = 1;

    for (size_t i = 0; i < n_pages; i += contig_pages_advanced) {
        auto const [noc_addr, contig_pages] =
            legacy_get_noc_addr_and_contiguous_pages_for_fabric_write<TENSOR_LAYOUT, MEM_LAYOUT>(
                curr_page_idx,
                offset_into_worker_slice,
                worker_slice_offset,
                tensor_addrgen,
                command_tensor.tensor_slice_shape);

        contig_pages_advanced = std::min<size_t>(contig_pages, n_pages);

        mcast_contig_pages_to_noc_address(
            noc0_dest_addr,
            l1_read_addr,
            contig_pages_advanced,
            payload_page_size,
            has_forward_fabric_connection,
            has_backward_fabric_connection,
            forward_fabric_sender,
            backward_fabric_sender,
            forward_direction_num_hops,
            backward_direction_num_hops);

        bool last_page_of_worker = ttnn::ccl::v2::advance_worker_global_page(
            curr_page_idx,
            offset_into_worker_slice,
            worker_slice_offset,
            worker_slice_shape.volume(),
            command_tensor.tensor_slice_shape,
            command_tensor.tensor_shape,
            contig_pages_advanced);

        noc_async_write_barrier();
        l1_read_addr += contig_pages_advanced * l1_scratch_page_size;
    }
}

//------------------------------------------------------------------------------
// Section 3: Local Read into L1 Scratchpad Functions
//------------------------------------------------------------------------------

template <typename AddrGen>
FORCE_INLINE void read_wrapped_chunk_from_output_tensor_to_address(
    uint32_t& curr_page_idx,
    uint32_t& offset_into_worker_slice,
    const ttnn::ccl::coord_t& offset_worker_slice,
    const ttnn::ccl::coord_t& worker_slice_shape,

    // In tiles for tile layout
    const ttnn::ccl::coord_t& tensor_shape,
    const ttnn::ccl::coord_t& tensor_slice_shape,
    const uint32_t local_l1_scratch_buffer_address,
    const AddrGen& s,
    const uint32_t num_pages,
    const uint32_t page_size,
    bool& last_page_of_worker) {
    // we expected caller to reset this and the last curr_page_idx when we set it true
    uint32_t local_l1_read_addr = local_l1_scratch_buffer_address;

    int32_t contig_pages = 1;
    for (uint32_t i = 0; i < num_pages; i += contig_pages) {
        contig_pages = 1;
#ifdef ROW_MAJOR_LAYOUT
#ifdef INTERLEAVED_MEM_LAYOUT
        uint64_t src_noc_addr = get_noc_addr(curr_page_idx, s);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size);
#elif defined SHARDED_MEM_LAYOUT
        ASSERT(false);  // unimplemented
#endif
#elif defined TILED_LAYOUT
#ifdef INTERLEAVED_MEM_LAYOUT
        noc_async_read_tile(curr_page_idx, s, local_l1_read_addr);
        // common with `write_chunk_v2`
#elif defined SHARDED_MEM_LAYOUT
        // TODO: Make d.get_noc_addr work on host + device
        auto const& [noc_yx, page_offset, contig_pages_] =
            s.get_page_location_with_contiguous_pages_in_row_in_bank(curr_page_idx);
        /*
         * num_pages - i: check if we are outside the number of pages remaining
         * contig_pages_: check if we are outside the max number of contig pages we can read in a row in a bank
         * contig_edge_of_tensor_slice: check if we are outside the edge of the tensor slice (in which case, we wrap
         * around if we aren't at the end)
         */
        uint32_t flattened_offset_worker_slice = offset_worker_slice.x + (offset_worker_slice.y * tensor_slice_shape.x);
        uint32_t contig_edge_of_tensor_slice =
            tensor_slice_shape.x - ((flattened_offset_worker_slice + offset_into_worker_slice) % tensor_slice_shape.x);

        contig_pages = std::min<int32_t>(num_pages - i, std::min<int32_t>(contig_pages_, contig_edge_of_tensor_slice));
        uint64_t src_noc_addr = get_noc_addr(
            static_cast<uint32_t>(noc_yx.noc_x), noc_yx.noc_y, s.bank_base_address + (page_offset * s.page_size) + 0);
        noc_async_read(src_noc_addr, local_l1_read_addr, page_size * contig_pages);
#endif

        // Update the curr_page_idx based on how the worker chunks + tensor slice is laid out in global tensor
        advance_worker_global_page_interleaved(
            curr_page_idx,  // Updated internally
            offset_into_worker_slice,
            offset_worker_slice,
            worker_slice_shape,
            tensor_slice_shape,
            tensor_shape,
            contig_pages,
            last_page_of_worker);

#endif
        local_l1_read_addr += page_size * contig_pages;
    }
    noc_async_read_barrier();
}

//------------------------------------------------------------------------------
// Section 4: Sync Signal Write Functions
//------------------------------------------------------------------------------

void mcast_sync_signal_to_addr(
    size_t some_buffering_addr,
    size_t& sync_details_arg_idx,
    bool has_forward_fabric_connection,
    bool has_backward_fabric_connection,
    tt::tt_fabric::WorkerToFabricEdmSender& forward_fabric_sender,
    tt::tt_fabric::WorkerToFabricEdmSender& backward_fabric_sender,
    size_t forward_direction_num_hops,
    size_t backward_direction_num_hops,
    size_t num_sync_signals) {
    auto send_sync_signal = [](size_t pkt_addr,
                               tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
                               size_t remote_sem_noc_x,
                               size_t remote_sem_noc_y,
                               size_t remote_sem_l1_addr,
                               size_t directional_num_hops) {
        static_assert(
            is_power_of_2(sizeof(PACKET_HEADER_TYPE)),
            "sizeof(tt::tt_fabric::PacketHeader) is not a power of two which violates the below assertion");
        ASSERT((pkt_addr & (sizeof(PACKET_HEADER_TYPE) - 1)) == 0);

        auto& pkt_hdr = *reinterpret_cast<PACKET_HEADER_TYPE*>(pkt_addr);
        pkt_hdr
            .to_chip_multicast(tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(directional_num_hops)})
            .to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                remote_sem_l1_addr,
                1,
                32,
                static_cast<uint8_t>(remote_sem_noc_x),
                static_cast<uint8_t>(remote_sem_noc_y)});
        fabric_connection.wait_for_empty_write_slot();
        fabric_connection.send_payload_flush_blocking_from_address(
            pkt_addr, pkt_hdr.get_payload_size_including_header());
    };

    for (size_t i = 0; i < num_sync_signals; ++i) {
        auto dest_sem_addr =
            get_arg_val<uint32_t>(sync_details_arg_idx++);  // hack, we pass in the address instead of the semaphore id
        auto dest_noc_x = get_arg_val<uint32_t>(sync_details_arg_idx++);
        auto dest_noc_y = get_arg_val<uint32_t>(sync_details_arg_idx++);

        if (has_forward_fabric_connection) {
            const size_t pkt_addr = some_buffering_addr;
            send_sync_signal(
                pkt_addr, forward_fabric_sender, dest_noc_x, dest_noc_y, dest_sem_addr, forward_direction_num_hops);
        }
        if (has_backward_fabric_connection) {
            const size_t pkt_addr = some_buffering_addr;
            send_sync_signal(
                pkt_addr, backward_fabric_sender, dest_noc_x, dest_noc_y, dest_sem_addr, backward_direction_num_hops);
        }

        auto sem_inc_noc_addr = get_noc_addr(dest_noc_x, dest_noc_y, dest_sem_addr);
        noc_semaphore_inc(sem_inc_noc_addr, 1);
    }
}

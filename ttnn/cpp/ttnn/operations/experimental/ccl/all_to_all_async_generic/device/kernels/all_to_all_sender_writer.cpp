#include "debug/dprint.h"
#include "dataflow_api.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "debug/waypoint.h"
#include <cstdint>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t current_device_id = get_compile_time_arg_val(1);
constexpr uint32_t ring_size = get_compile_time_arg_val(2);
constexpr uint32_t outer_dims_size = get_compile_time_arg_val(3);
constexpr uint32_t concat_dim_size = get_compile_time_arg_val(4);
constexpr uint32_t inner_dims_size = get_compile_time_arg_val(5);
constexpr uint32_t concat_block_size = get_compile_time_arg_val(6);
constexpr uint32_t has_half_tile = get_compile_time_arg_val(7);
constexpr uint32_t output_page_size = get_compile_time_arg_val(8);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(9);

template <typename AddrGenType>
void write_data(
    bool last,
    bool local,
    uint32_t dest_id,
    AddrGenType addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr,
    tt::tt_fabric::WorkerToFabricEdmSender& fabric_connection,
    size_t l1_read_addr,
    uint32_t payload_size_bytes,
    uint32_t offset,
    uint64_t output_semaphore_noc_addr_in_pkt,
    uint64_t output_semaphore_noc_addr_in_pkt_local,
    uint32_t device_id) {
    if (last) {
        if (local) {
            noc_semaphore_inc(output_semaphore_noc_addr_in_pkt_local, 1);
            noc_async_write(l1_read_addr, addrgen.get_noc_addr(dest_id, offset), payload_size_bytes);
            noc_async_write_barrier();
        } else {
            perform_atomic_fabric_write(
                pkt_hdr,
                dest_id,
                addrgen,
                fabric_connection,
                l1_read_addr,
                payload_size_bytes,
                output_semaphore_noc_addr_in_pkt,
                1,
                32,
                false,
                offset);
        }
    } else {
        if (local) {
            noc_async_write(l1_read_addr, addrgen.get_noc_addr(dest_id, offset), payload_size_bytes);
        } else {
            tt::tt_fabric::linear::to_noc_unicast_write(payload_size_bytes, pkt_hdr, dest_id, addrgen, offset);
            perform_payload_send(fabric_connection, l1_read_addr, payload_size_bytes, pkt_hdr);
        }
    }
    noc_async_writes_flushed();
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    size_t arg_idx = 0;
    address_t output_address = get_arg_val<address_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_core_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_core_y = get_arg_val<uint32_t>(arg_idx++);
    constexpr auto output_args = TensorAccessorArgs<10>();
    auto output_addrgen = TensorAccessor(output_args, output_address, output_page_size);

    // Build fabric connection
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);
    uint64_t output_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(receiver_core_x, receiver_core_y, global_semaphore_addr, 0);
    uint64_t output_semaphore_noc_addr_in_pkt_local =
        safe_get_noc_addr(receiver_core_x, receiver_core_y, global_semaphore_addr);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);

    fabric_connection.open_finish();

    constexpr uint32_t padded_page_size = output_page_size / 2;
    constexpr uint32_t full_block_offset = (has_half_tile)
                                               ? concat_block_size * current_device_id + (current_device_id + 1) / 2
                                               : concat_block_size * current_device_id;
    constexpr uint32_t pre_block_offset = full_block_offset - 1;
    constexpr uint32_t post_block_offset = full_block_offset + concat_block_size;
    constexpr bool has_pre_half_tile = has_half_tile && current_device_id % 2 == 1;
    constexpr bool has_full = concat_block_size != 0;
    constexpr bool has_post_half_tile = has_half_tile && current_device_id % 2 == 0;

    for (uint32_t device_id = 0; device_id < ring_size; ++device_id) {
        volatile PACKET_HEADER_TYPE* pkt_hdr;
        tt::tt_fabric::WorkerToFabricEdmSender& cur_connection = (device_id > current_device_id)
                                                                     ? fabric_connection.get_forward_connection()
                                                                     : fabric_connection.get_backward_connection();
        if (device_id > current_device_id) {
            pkt_hdr = pkt_hdr_forward;
            pkt_hdr->to_chip_unicast(device_id - current_device_id);
        } else {
            pkt_hdr = pkt_hdr_backward;
            pkt_hdr->to_chip_unicast(current_device_id - device_id);
        }

        for (uint32_t o = 0; o < outer_dims_size; ++o) {
            // Write pre half-sized tiles
            if (has_pre_half_tile) {
                for (uint32_t i = 0; i < inner_dims_size; ++i) {
                    const uint32_t dest_tile_id =
                        o * inner_dims_size * concat_dim_size + pre_block_offset * inner_dims_size + i;

                    cb_wait_front(cb0_id, 1);
                    size_t l1_read_addr = get_read_ptr(cb0_id);
                    bool last = (!has_full && o == outer_dims_size - 1 && i == inner_dims_size - 1);
                    write_data(
                        last,
                        device_id == current_device_id,
                        dest_tile_id,
                        output_addrgen,
                        pkt_hdr,
                        cur_connection,
                        l1_read_addr,
                        padded_page_size,
                        (current_device_id % 2) * padded_page_size,
                        output_semaphore_noc_addr_in_pkt,
                        output_semaphore_noc_addr_in_pkt_local,
                        device_id);
                    cb_pop_front(cb0_id, 1);
                }
            }

            // Write main full tiles
            for (uint32_t c = 0; c < concat_block_size; ++c) {
                for (uint32_t i = 0; i < inner_dims_size; ++i) {
                    const uint32_t dest_tile_id =
                        o * inner_dims_size * concat_dim_size + (c + full_block_offset) * inner_dims_size + i;

                    cb_wait_front(cb0_id, 1);
                    size_t l1_read_addr = get_read_ptr(cb0_id);
                    bool last =
                        (!has_post_half_tile && o == outer_dims_size - 1 && c == concat_block_size - 1 &&
                         i == inner_dims_size - 1);
                    write_data(
                        last,
                        device_id == current_device_id,
                        dest_tile_id,
                        output_addrgen,
                        pkt_hdr,
                        cur_connection,
                        l1_read_addr,
                        output_page_size,
                        0,
                        output_semaphore_noc_addr_in_pkt,
                        output_semaphore_noc_addr_in_pkt_local,
                        device_id);
                    cb_pop_front(cb0_id, 1);
                }
            }

            // Write post half-sized tiles
            if (has_post_half_tile) {
                for (uint32_t i = 0; i < inner_dims_size; ++i) {
                    const uint32_t dest_tile_id =
                        o * inner_dims_size * concat_dim_size + post_block_offset * inner_dims_size + i;

                    cb_wait_front(cb0_id, 1);
                    size_t l1_read_addr = get_read_ptr(cb0_id);
                    bool last = (o == outer_dims_size - 1 && i == inner_dims_size - 1);
                    write_data(
                        last,
                        device_id == current_device_id,
                        dest_tile_id,
                        output_addrgen,
                        pkt_hdr,
                        cur_connection,
                        l1_read_addr,
                        padded_page_size,
                        0,
                        output_semaphore_noc_addr_in_pkt,
                        output_semaphore_noc_addr_in_pkt_local,
                        device_id);
                    cb_pop_front(cb0_id, 1);
                }
            }
        }
    }

    fabric_connection.close();
}

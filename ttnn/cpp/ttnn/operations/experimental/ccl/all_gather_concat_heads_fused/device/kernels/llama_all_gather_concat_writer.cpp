// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include <array>

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 1;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_core = 4;  // get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = 19;  // get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = 18;  // get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = 12;   // get_arg_val<uint32_t>(arg_idx++);

    const uint32_t concat_semaphore_send_addr = get_semaphore(get_arg_val<uint32_t>(arg_idx++));

    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;

    tt_l1_ptr uint32_t* mcast_dest_noc_start_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += 3;
    tt_l1_ptr uint32_t* mcast_dest_noc_start_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += 3;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += 3;
    tt_l1_ptr uint32_t* mcast_dest_noc_end_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += 3;

    size_t arg_for_fab = arg_idx;
    auto fabric_connection =
        FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
            arg_idx);

    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "reserved_packet_header_cb_id: " << (uint32_t)reserved_packet_header_cb_id << "\n";
    DPRINT << "num_packet_headers_storable: " << (uint32_t)num_packet_headers_storable << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "num_targets_forward_direction: " << (uint32_t)num_targets_forward_direction << "\n";
    DPRINT << "num_targets_backward_direction: " << (uint32_t)num_targets_backward_direction << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "num_tiles_per_core: " << (uint32_t)num_tiles_per_core << "\n";
    DPRINT << "num_tiles_to_read: " << (uint32_t)num_tiles_to_read << "\n";
    DPRINT << "first_core_tile_start_offset: " << (uint32_t)first_core_tile_start_offset << "\n";
    DPRINT << "num_cores: " << (uint32_t)num_cores << "\n";
    for (uint32_t i = 0; i < num_cores; i++) {
        DPRINT << "core_noc_x[" << i << "]: " << (uint32_t)core_noc_x[i] << "\n";
        DPRINT << "core_noc_y[" << i << "]: " << (uint32_t)core_noc_y[i] << "\n";
    }
    DPRINT << "wait_output_semaphore: " << (uint32_t)wait_output_semaphore << "\n";
    DPRINT << "reset_global_semaphore: " << (uint32_t)reset_global_semaphore << "\n";
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << "\n";

    DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

    uint32_t concat_arg_start = get_arg_val<uint32_t>(0);
    uint32_t in_tile_offset_by_head = get_arg_val<uint32_t>(concat_arg_start);
    uint32_t q_start_addr = get_arg_val<uint32_t>(concat_arg_start + 1);

    constexpr uint32_t ELEMENT_SIZE = get_compile_time_arg_val(8);
    constexpr uint32_t SUBTILE_LINE_BYTES = get_compile_time_arg_val(9);
    constexpr uint32_t cb_id_q_out = get_compile_time_arg_val(10);
    constexpr uint32_t head_size = get_compile_time_arg_val(11);
    constexpr uint32_t batch = get_compile_time_arg_val(12);
    constexpr uint32_t head_size_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t PHASES_TO_READ =
        get_compile_time_arg_val(14);  // 0 to read all phases, 1 to read only first phase, 2 to read only second phase

    constexpr uint32_t in_num_cores = get_compile_time_arg_val(15);
    constexpr uint32_t face_h = get_compile_time_arg_val(16);
    constexpr uint32_t face_hw = get_compile_time_arg_val(17);

    constexpr uint32_t temp_cb_id = get_compile_time_arg_val(18);

    constexpr uint32_t batch_size = get_compile_time_arg_val(19);
    constexpr uint32_t batch_start_1 = get_compile_time_arg_val(20);
    constexpr uint32_t batch_end_1 = get_compile_time_arg_val(21);
    constexpr uint32_t batch_start_2 = get_compile_time_arg_val(22);
    constexpr uint32_t batch_end_2 = get_compile_time_arg_val(23);
    constexpr uint32_t start_local = get_compile_time_arg_val(24);

    auto batch_loop = [&](uint32_t head_size_num_tiles,
                          uint32_t q_start_addr,
                          uint32_t face_h,
                          uint32_t SUBTILE_LINE_BYTES,
                          uint32_t face_hw,
                          uint32_t ELEMENT_SIZE,
                          const uint32_t cb_write_ptr_base,
                          uint64_t qkv_read_addr,
                          uint32_t tile_size,
                          uint32_t num_tiles_read_cur_core,
                          uint32_t cur_core_idx,
                          uint32_t num_tiles_per_core_concat,
                          uint32_t start,
                          uint32_t end,
                          uint32_t concat_arg_start) {
        tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + concat_arg_start));
        tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + in_num_cores + concat_arg_start));

        for (uint32_t q = start; q < end; ++q) {
            uint32_t wptr_offset = q < face_h ? q * SUBTILE_LINE_BYTES : (q + face_h) * SUBTILE_LINE_BYTES;
            uint32_t q_write_addr = cb_write_ptr_base + wptr_offset;
            for (uint32_t i = 0; i < head_size_num_tiles; ++i) {
                noc_async_read(
                    qkv_read_addr + face_hw * ELEMENT_SIZE, q_write_addr + face_hw * ELEMENT_SIZE, SUBTILE_LINE_BYTES);

                qkv_read_addr += tile_size;
                q_write_addr += tile_size;
                num_tiles_read_cur_core++;

                if (num_tiles_read_cur_core == num_tiles_per_core_concat) {
                    cur_core_idx++;
                    qkv_read_addr =
                        get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                        in_tile_offset_by_head;
                    num_tiles_read_cur_core = 0;
                }
            }
        }
    };

    auto nlp_concat = [&](uint32_t head_size_num_tiles,
                          uint32_t batch,
                          uint32_t q_start_addr,
                          uint32_t head_size,
                          uint32_t cb_id_q_out,
                          uint32_t face_h,
                          uint32_t SUBTILE_LINE_BYTES,
                          uint32_t face_hw,
                          uint32_t ELEMENT_SIZE,
                          uint32_t concat_arg_start,
                          bool nlp_local,
                          uint32_t start_local) {
        tt_l1_ptr uint32_t* in0_mcast_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + concat_arg_start));
        tt_l1_ptr uint32_t* in0_mcast_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(2 + in_num_cores + concat_arg_start));

        // Q
        uint32_t cur_core_idx = batch_start_1;
        uint32_t total_input_cores = in_num_cores;
        uint32_t num_tiles_per_core_concat = (head_size_num_tiles * batch) / total_input_cores;

        uint64_t qkv_read_addr =
            get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
            in_tile_offset_by_head;

        uint32_t num_tiles_read_cur_core = 0;
        uint32_t q_write_addr = 0;
        uint32_t tile_size = head_size / head_size_num_tiles;
        const uint32_t cb_write_ptr_base = get_write_ptr(cb_id_q_out);

        uint32_t start = nlp_local ? start_local : batch_start_1;
        uint32_t end = nlp_local ? start_local + 8 : batch_end_1;
        uint32_t idx_end = nlp_local ? 1 : batch_size;

        for (uint32_t batch_range = 0; batch_range < idx_end; batch_range++) {
            batch_loop(
                head_size_num_tiles,
                q_start_addr,
                face_h,
                SUBTILE_LINE_BYTES,
                face_hw,
                ELEMENT_SIZE,
                cb_write_ptr_base,
                qkv_read_addr,
                tile_size,
                num_tiles_read_cur_core,
                cur_core_idx,
                num_tiles_per_core_concat,
                start,
                end,
                concat_arg_start);
            start = batch_start_2;
            end = batch_end_2;
            cur_core_idx = batch_start_2;
            qkv_read_addr = get_noc_addr(in0_mcast_noc_x[cur_core_idx], in0_mcast_noc_y[cur_core_idx], q_start_addr) +
                            in_tile_offset_by_head;
        }

        noc_async_read_barrier();
    };

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);

    DPRINT << "packet_header_buffer_addr_forward: " << (uint32_t)packet_header_buffer_addr_forward << "\n";
    DPRINT << "packet_header_buffer_addr_backward: " << (uint32_t)packet_header_buffer_addr_backward << "\n";

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    pkt_hdr_backward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = first_core_tile_start_offset;
    uint32_t core_id = 0;
    while (tiles_read < num_tiles_to_read) {
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = std::min(num_tiles_to_read - tiles_read, num_tiles_to_read_this_core);
        cb_wait_front(cb0_id, num_tiles_to_read_this_core);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint64_t noc0_dest_noc_addr = get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0, 0);
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            num_tiles_to_read_this_core * tensor0_page_size);

        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }

        noc_async_writes_flushed();
        cb_pop_front(cb0_id, num_tiles_to_read_this_core);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        auto* pkt_hdr_fwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
        pkt_hdr_fwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});

        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr_fwd->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
            reinterpret_cast<uint32_t>(pkt_hdr_fwd), sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        auto* pkt_hdr_bwd = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
        pkt_hdr_bwd->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});

        pkt_hdr_bwd->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            reinterpret_cast<uint32_t>(pkt_hdr_bwd), sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    DPRINT << "inc done\n";

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) < out_ready_sem_wait_value);
        DPRINT << "waitval done\n";
    }

    // Set up for mcasting to concat workers
    if (wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* concat_semaphore_send_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(concat_semaphore_send_addr);
        noc_semaphore_set(concat_semaphore_send_addr_ptr, VALID);
    }

    if (wait_output_semaphore) {
        for (uint32_t i = 0; i < 3; i++) {
            const uint64_t concat_sem_rcv_addr = get_noc_multicast_addr(
                mcast_dest_noc_start_x[i],
                mcast_dest_noc_start_y[i],
                mcast_dest_noc_end_x[i],
                mcast_dest_noc_end_y[i],
                concat_semaphore_send_addr);
            noc_semaphore_set_multicast(
                concat_semaphore_send_addr,
                concat_sem_rcv_addr,
                i == 1 ? 3 : 2,
                false,  // linked = false
                true);  // multicast_path_reserve = true
        }
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_start();
    }

    if (!wait_output_semaphore) {
        volatile tt_l1_ptr uint32_t* signal_semaphore_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(concat_semaphore_send_addr);
        noc_semaphore_wait(signal_semaphore_addr_ptr, VALID);
        // noc_semaphore_set(signal_semaphore_addr_ptr, 0);
    }

    DPRINT << "START CONCAT HEADS\n";

    nlp_concat(
        head_size_num_tiles,
        batch,
        q_start_addr,
        head_size,
        cb_id_q_out,
        face_h,
        SUBTILE_LINE_BYTES,
        face_hw,
        ELEMENT_SIZE,
        concat_arg_start,
        0,
        start_local);

    if (reset_global_semaphore) {
        const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr);
        noc_inline_dw_write(dest_noc_addr, 0);
        DPRINT << "reset done\n";
    }

    // noc_async_read_barrier();
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close_finish();
    }
    noc_async_write_barrier();

    DPRINT << "DONE\n";
}

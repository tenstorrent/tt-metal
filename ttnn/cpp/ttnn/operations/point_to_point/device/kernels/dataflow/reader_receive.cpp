#include "dataflow_api.h"

#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

using tt::data_movement::common::round_up;
using tt::data_movement::common::tt_memmove;

void kernel_main() {
    constexpr bool intermediate_is_dram = get_compile_time_arg_val(0);
    constexpr uint32_t packet_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t receiver_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t alignment = get_compile_time_arg_val(3);

    const auto page_idx_start = get_arg_val<uint32_t>(0);
    const auto page_idx_end = get_arg_val<uint32_t>(1);

    const auto max_pages_per_packet = get_arg_val<uint32_t>(2);

    const auto intermediate_base_addr = get_arg_val<uint32_t>(3);
    const auto packet_size_bytes = get_arg_val<uint32_t>(4);
    const auto page_size_bytes = get_arg_val<uint32_t>(5);
    volatile tt_l1_ptr uint32_t* semaphore_ptr = get_arg_val<volatile tt_l1_ptr uint32_t*>(6);

    const uint32_t aligned_page_size_bytes = round_up(page_size_bytes, alignment);

    InterleavedAddrGenFast<intermediate_is_dram> packet_buffer_addrgen{
        .bank_base_address = intermediate_base_addr,
        .page_size = packet_size_bytes,
        .data_format = get_dataformat(packet_cb_id)};

    cb_reserve_back(packet_cb_id, 1);
    const uint64_t packet_l1_addr = get_write_ptr(packet_cb_id);
    cb_push_back(packet_cb_id, 1);

    DPRINT << "SEMAPHORE WAIT addr: " << (uint32_t)semaphore_ptr << "\n";
    noc_semaphore_wait(semaphore_ptr, 1);
    DPRINT << "SEMAPHORE DONE" << "\n";

    uint32_t packet_idx = page_idx_start / max_pages_per_packet;
    uint32_t packet_page_count = page_idx_start % max_pages_per_packet;
    for (uint32_t page_idx = page_idx_start; page_idx < page_idx_end; ++page_idx, ++packet_page_count) {
        if (page_idx == page_idx_start || packet_page_count == max_pages_per_packet || page_idx == page_idx_end - 1) {
            DPRINT << "READ PACKET idx: " << packet_idx << "\n";
            const uint64_t packet_noc_addr = packet_buffer_addrgen.get_noc_addr(packet_idx);
            noc_async_read(packet_noc_addr, packet_l1_addr, packet_size_bytes);
            noc_async_read_barrier();
            DPRINT << "PACKET READ DONE" << packet_idx << "\n";
            packet_page_count = 0;
        }

        DPRINT << "RESERVE BACK page idx: " << page_idx << "\n";
        cb_reserve_back(receiver_cb_id, 1);
        const uint32_t page_l1_addr = get_write_ptr(receiver_cb_id);
        uint32_t packet_l1_page_addr = packet_l1_addr + packet_page_count * aligned_page_size_bytes;
        tt_memmove<true, true, true, 0>(page_l1_addr, packet_l1_page_addr, page_size_bytes);
        DPRINT << "PACKET PAGE CB DONE " << "\n";
        cb_push_back(receiver_cb_id, 1);
    }
    // TODO BETTER ASYNC

    DPRINT << "KERNEL DONE" << "\n";
}

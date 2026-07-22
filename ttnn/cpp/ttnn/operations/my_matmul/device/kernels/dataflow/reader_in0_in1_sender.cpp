#include <stdint.h>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t src1_addr = get_arg_val<uint32_t>(1);

    uint32_t Mt = get_arg_val<uint32_t>(2);
    uint32_t Kt = get_arg_val<uint32_t>(3);
    uint32_t Nt = get_arg_val<uint32_t>(4);
    uint32_t top = get_arg_val<uint32_t>(5);
    uint32_t left = get_arg_val<uint32_t>(6);
    uint32_t bot = get_arg_val<uint32_t>(7);
    uint32_t right = get_arg_val<uint32_t>(8);
    uint32_t num_blocks_k = get_arg_val<uint32_t>(9);
    uint32_t sub_block_k = get_arg_val<uint32_t>(10);
    uint32_t k_block_A = get_arg_val<uint32_t>(11);
    uint32_t k_block_B = get_arg_val<uint32_t>(12);

    uint32_t in0_recv_start_x = get_arg_val<uint32_t>(13);
    uint32_t in0_recv_start_y = get_arg_val<uint32_t>(14);
    uint32_t in0_recv_end_x = get_arg_val<uint32_t>(15);
    uint32_t in0_recv_end_y = get_arg_val<uint32_t>(16);
    uint32_t in0_sender_phys_x = get_arg_val<uint32_t>(17);
    uint32_t in0_sender_phys_y = get_arg_val<uint32_t>(18);

    uint32_t in1_recv_start_x = get_arg_val<uint32_t>(19);
    uint32_t in1_recv_start_y = get_arg_val<uint32_t>(20);
    uint32_t in1_recv_end_x = get_arg_val<uint32_t>(21);
    uint32_t in1_recv_end_y = get_arg_val<uint32_t>(22);
    uint32_t in1_sender_phys_x = get_arg_val<uint32_t>(23);
    uint32_t in1_sender_phys_y = get_arg_val<uint32_t>(24);

    uint32_t in0_num_dests = get_arg_val<uint32_t>(25);
    uint32_t in1_num_dests = get_arg_val<uint32_t>(26);

    constexpr uint32_t cb_in0 = 0;
    constexpr uint32_t cb_in1 = 1;

    constexpr auto s0_args = TensorAccessorArgs<0>();
    const auto s0 = TensorAccessor(s0_args, src0_addr);
    constexpr auto s1_args = TensorAccessorArgs<s0_args.next_compile_time_args_offset()>();
    const auto s1 = TensorAccessor(s1_args, src1_addr);

    const uint32_t in0_tile_bytes = get_tile_size(cb_in0);
    const uint32_t in1_tile_bytes = get_tile_size(cb_in1);

    const uint32_t in0_multicast_bytes = k_block_A * get_tile_size(cb_in0);
    const uint32_t in1_multicast_bytes = k_block_B * get_tile_size(cb_in1);

    uint32_t in0_sender_sem_addr = get_semaphore(0);
    uint32_t in0_receiver_sem_addr = get_semaphore(1);
    uint32_t in1_sender_sem_addr = get_semaphore(2);
    uint32_t in1_receiver_sem_addr = get_semaphore(3);
    volatile tt_l1_ptr uint32_t* in0_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_sender_sem_addr);
    volatile tt_l1_ptr uint32_t* in0_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in0_receiver_sem_addr);
    volatile tt_l1_ptr uint32_t* in1_sender_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_sender_sem_addr);
    volatile tt_l1_ptr uint32_t* in1_receiver_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(in1_receiver_sem_addr);

    noc_semaphore_set(in0_receiver_sem_ptr, VALID);  // set local receiver flag to VALID - this value is later multicast
                                                     // by noc_semaphore_set_multicast() calls
    noc_semaphore_set(in1_receiver_sem_ptr, VALID);

    for (uint32_t block_iter_k = 0; block_iter_k < num_blocks_k; block_iter_k++) {
        cb_reserve_back(cb_in0, k_block_A);
        cb_reserve_back(cb_in1, k_block_B);

        uint32_t k_offset = block_iter_k * sub_block_k;
        uint32_t in0_l1_write_addr = get_write_ptr(cb_in0);
        uint32_t in0_block_start_l1_write_adder = in0_l1_write_addr;  // save the addr for later multicast
        for (uint32_t y = top; y < bot; y++) {
            for (uint32_t kt = k_offset; kt < k_offset + sub_block_k; kt++) {
                noc_async_read_page(y * Kt + kt, s0, in0_l1_write_addr);
                in0_l1_write_addr += in0_tile_bytes;
            }
        }

        uint32_t in1_l1_write_addr = get_write_ptr(cb_in1);
        uint32_t in1_block_start_l1_write_adder = in1_l1_write_addr;  // save the addr for later multicast
        for (uint32_t kt = k_offset; kt < k_offset + sub_block_k; kt++) {
            const uint32_t row = kt * Nt;
            for (uint32_t x = left; x < right; x++) {
                noc_async_read_page(row + x, s1, in1_l1_write_addr);
                in1_l1_write_addr += in1_tile_bytes;
            }
        }

        // Wait for data to arrive...
        noc_async_read_barrier();

        noc_semaphore_wait(in0_sender_sem_ptr, in0_num_dests);  // wait for receivers to signal ready
        noc_semaphore_set(in0_sender_sem_ptr, 0);               // reset for next iteration

        noc_semaphore_wait(in1_sender_sem_ptr, in1_num_dests);  // wait for receivers to signal ready
        noc_semaphore_set(in1_sender_sem_ptr, 0);               // reset for next iteration

        uint64_t in0_mcast = get_noc_multicast_addr(
            in0_recv_start_x, in0_recv_start_y, in0_recv_end_x, in0_recv_end_y, in0_block_start_l1_write_adder);
        noc_async_write_multicast(
            in0_block_start_l1_write_adder,
            in0_mcast,
            in0_multicast_bytes,
            in0_num_dests);  // data to all receivers' L1
        uint64_t in0_sem_mcast = get_noc_multicast_addr(
            in0_recv_start_x, in0_recv_start_y, in0_recv_end_x, in0_recv_end_y, in0_receiver_sem_addr);
        noc_semaphore_set_multicast(in0_receiver_sem_addr, in0_sem_mcast, in0_num_dests);  // flag VALID to all

        uint64_t in1_mcast = get_noc_multicast_addr(
            in1_recv_start_x, in1_recv_start_y, in1_recv_end_x, in1_recv_end_y, in1_block_start_l1_write_adder);
        noc_async_write_multicast(
            in1_block_start_l1_write_adder,
            in1_mcast,
            in1_multicast_bytes,
            in1_num_dests);  // data to all receivers' L1
        uint64_t in1_sem_mcast = get_noc_multicast_addr(
            in1_recv_start_x, in1_recv_start_y, in1_recv_end_x, in1_recv_end_y, in1_receiver_sem_addr);
        noc_semaphore_set_multicast(in1_receiver_sem_addr, in1_sem_mcast, in1_num_dests);  // flag VALID to all

        cb_push_back(cb_in0, k_block_A);
        cb_push_back(cb_in1, k_block_B);
    }
}

// // TODO: fix block sizes
// cb_reserve_back(cb_id_in0, block_size_A);
// uint32_t in0_l1_write_addr = get_write_ptr(cb_id_in0);
// const uint32_t in0_tile_bytes = get_tile_size(cb_id_in0);
// for (uint32_t block_iter_m = top; block_iter_m < bot; block_iter_m += sub_block_m) {
//     for (uint32_t sub_block_iter_m = block_iter_m; sub_block_iter_m < block_iter_m + sub_block_m; block_iter_m++) {
//         const uint32_t row = sub_block_iter_m * Kt;
//         for (int kt = 0; kt < Kt; kt++)
//         {
//             noc_async_read_page(row + kt, s0, in0_l1_write_addr);
//             in0_l1_write_addr += in0_tile_bytes;
//         }
//     }
// }

// cb_reserve_back(cb_id_in1, block_size_B);
// uint32_t in1_l1_write_addr = get_write_ptr(cb_id_in1);
// const uint32_t in1_tile_bytes = get_tile_size(cb_id_in1);
// for (uint32_t kt = 0; kt < Kt; kt++) {
//     for(uint32_t block_iter_n = left; block_iter_n < right; block_iter_n += sub_block_n) {
//         const uint32_t row = kt * Nt;
//         for (uint32_t subblock_iter_n = block_iter_n; subblock_iter_n < block_iter_n + sub_block_n;
//         subblock_iter_n++)
//         {
//             noc_async_read_page(row + sub_block_iter_n, s1, in1_l1_write_addr);
//             in1_l1_write_addr += in1_tile_bytes;
//         }
//     }
// }

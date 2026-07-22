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

    const uint32_t multicast_bytes = k_block_A * get_tile_size(cb_in0);

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

    for (uint32_t block_iter_k = 0; block_iter_k < num_blocks_k; block_iter_k++) {
        cb_reserve_back(cb_in0, k_block_A);
        cb_reserve_back(cb_in1, k_block_B);

        // Read in0 sender's data
        uint64_t in0_sender_counter = get_noc_addr(in0_sender_phys_x, in0_sender_phys_y, in0_sender_sem_addr);
        noc_semaphore_set(in0_receiver_sem_ptr, INVALID);  // reinit semaphore to INVALID
        noc_semaphore_inc(in0_sender_counter, 1);          // signal to sender semaphore we are ready
        noc_semaphore_wait(in0_receiver_sem_ptr, VALID);   // wait until sender increments our semaphore

        // Read in1 sender's data
        uint64_t in1_sender_counter = get_noc_addr(in1_sender_phys_x, in1_sender_phys_y, in1_sender_sem_addr);
        noc_semaphore_set(in1_receiver_sem_ptr, INVALID);  // reinit semaphore to INVALID
        noc_semaphore_inc(in1_sender_counter, 1);          // signal to sender semaphore we are ready
        noc_semaphore_wait(in1_receiver_sem_ptr, VALID);   // wait until sender increments our semaphore

        // Is this needed?
        // noc_async_read_barrier();

        cb_push_back(cb_in0, k_block_A);
        cb_push_back(cb_in1, k_block_B);
    }
}

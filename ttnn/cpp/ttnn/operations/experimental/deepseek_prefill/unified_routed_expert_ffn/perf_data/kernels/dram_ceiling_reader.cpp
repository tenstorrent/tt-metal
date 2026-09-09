// DRAM read-bandwidth ceiling microbenchmark kernel (scratch, not for commit).
//
// CT args:
//   0 mode        0 = interleaved page reads via TensorAccessor (page_id stride pattern)
//                 1 = bank-direct contiguous bursts (get_noc_addr_from_bank_id), xfer bytes each
//   1 xfer_bytes  bytes per NoC transaction (mode 0: must equal the tensor page size; mode 1: burst size <= 16384)
//   2 group       reads issued per transaction-id group before rotating trids (in-flight depth ~= 2*group)
//   3 ring_bytes  size of the L1 scratch ring (CB 0)
//   4.. TensorAccessorArgs (mode 0 only; still appended for mode 1 but unused)
// RT args:
//   0 buffer_addr  1 num_reads  2 first (mode0: first page id; mode1: first byte offset in bank)
//   3 stride       (mode0: page id stride; mode1: byte stride between bursts)   4 bank_id  5 vc
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t mode = get_compile_time_arg_val(0);
    constexpr uint32_t xfer_bytes = get_compile_time_arg_val(1);
    constexpr uint32_t group = get_compile_time_arg_val(2);
    constexpr uint32_t ring_bytes = get_compile_time_arg_val(3);

    const uint32_t buffer_addr = get_arg_val<uint32_t>(0);
    const uint32_t num_reads = get_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(2);
    const uint32_t stride = get_arg_val<uint32_t>(3);
    const uint32_t bank_id = get_arg_val<uint32_t>(4);
    const uint32_t vc = get_arg_val<uint32_t>(5);

    if (num_reads == 0) {
        return;
    }

    cb_reserve_back(0, 1);
    const uint32_t l1_base = get_write_ptr(0);
    const uint32_t l1_end = l1_base + ring_bytes;
    uint32_t l1w = l1_base;

    // 3 rotating transaction ids; wait on the oldest once two groups are in flight (reader_dram.cpp pattern).
    constexpr uint32_t kNumTrid = 3;
    uint32_t cur_trid = 1;
    uint32_t wait_trid = 1;
    uint32_t groups_in_flight = 0;

    if constexpr (mode == 0) {
        constexpr auto ta_args = TensorAccessorArgs<4>();
        const auto acc = TensorAccessor(ta_args, buffer_addr, xfer_bytes);
        uint32_t pid = first;
        uint32_t in_group = 0;
        noc_async_read_set_trid(cur_trid);
        for (uint32_t i = 0; i < num_reads; ++i) {
            noc_async_read_page(pid, acc, l1w);
            pid += stride;
            l1w += xfer_bytes;
            if (l1w + xfer_bytes > l1_end) {
                l1w = l1_base;
            }
            if (++in_group == group) {
                in_group = 0;
                if (groups_in_flight == 2) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    wait_trid = (wait_trid == kNumTrid) ? 1 : wait_trid + 1;
                } else {
                    ++groups_in_flight;
                }
                cur_trid = (cur_trid == kNumTrid) ? 1 : cur_trid + 1;
                noc_async_read_set_trid(cur_trid);
            }
        }
    } else {
        const uint64_t src_base = get_noc_addr_from_bank_id<true>(bank_id, buffer_addr);
        noc_async_read_one_packet_set_state<true>(src_base, xfer_bytes, vc);
        uint32_t off = first;
        uint32_t in_group = 0;
        noc_async_read_set_trid(cur_trid);
        for (uint32_t i = 0; i < num_reads; ++i) {
            noc_async_read_one_packet_with_state_with_trid(src_base, off, l1w, cur_trid);
            off += stride;
            l1w += xfer_bytes;
            if (l1w + xfer_bytes > l1_end) {
                l1w = l1_base;
            }
            if (++in_group == group) {
                in_group = 0;
                if (groups_in_flight == 2) {
                    noc_async_read_barrier_with_trid(wait_trid);
                    wait_trid = (wait_trid == kNumTrid) ? 1 : wait_trid + 1;
                } else {
                    ++groups_in_flight;
                }
                cur_trid = (cur_trid == kNumTrid) ? 1 : cur_trid + 1;
            }
        }
    }
    noc_async_read_barrier();
}

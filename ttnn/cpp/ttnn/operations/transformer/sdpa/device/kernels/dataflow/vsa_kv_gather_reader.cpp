// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// vsa_ring_sdpa's K/V ring all-gather, sender reader (forked from all_gather_async's minimal_default_reader).
// See vsa_kv_gather_writer.cpp for the protocol. This side (a) pushes the local slice's tiles to the writer in the
// token-major order, (b) for every slice that lands from the other neighbour waits on out_ready_sem per packet group
// and, while more forwarding is due, re-reads the landed tiles from the gathered buffers for the writer to forward,
// (c) signals the fused op per landed slice.

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////
constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t cb_output_id = get_compile_time_arg_val(2);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t page_size = get_compile_time_arg_val(4);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(6);
constexpr uint32_t n_heads = get_compile_time_arg_val(7);
constexpr uint32_t dht = get_compile_time_arg_val(8);
constexpr uint32_t ht_local = get_compile_time_arg_val(9);
constexpr uint32_t ht_total = get_compile_time_arg_val(10);
constexpr bool fuse_op = get_compile_time_arg_val(11);
constexpr uint32_t accessor_args_base = 12;
constexpr auto k_args = TensorAccessorArgs<accessor_args_base>();
constexpr auto v_args = TensorAccessorArgs<k_args.next_compile_time_args_offset()>();
constexpr auto gk_args = TensorAccessorArgs<v_args.next_compile_time_args_offset()>();
constexpr auto gv_args = TensorAccessorArgs<gk_args.next_compile_time_args_offset()>();

constexpr uint32_t tiles_per_row = 2 * n_heads * dht;

struct SeqCursor {
    uint32_t r = 0, t = 0, h = 0, c = 0;
    void advance() {
        if (++c == dht) {
            c = 0;
            if (++h == n_heads) {
                h = 0;
                if (++t == 2) {
                    t = 0;
                    ++r;
                }
            }
        }
    }
    uint32_t local_page() const { return h * ht_local * dht + r * dht + c; }
    uint32_t gathered_page(uint32_t chip) const { return h * ht_total * dht + (chip * ht_local + r) * dht + c; }
};

void kernel_main() {
    uint32_t arg_idx = 0;
    const address_t k_address = get_arg_val<address_t>(arg_idx++);
    const address_t v_address = get_arg_val<address_t>(arg_idx++);
    const address_t gk_address = get_arg_val<address_t>(arg_idx++);
    const address_t gv_address = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem = get_arg_val<uint32_t>(arg_idx++);
    const bool direction = get_arg_val<uint32_t>(arg_idx++);  // 0: receives from the backward neighbour, 1: forward
    const uint32_t row_start = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t row_end = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t chunks_per_sync = get_arg_val<uint32_t>(arg_idx++);

    const auto k = TensorAccessor(k_args, k_address);
    const auto v = TensorAccessor(v_args, v_address);
    const auto gk = TensorAccessor(gk_args, gk_address);
    const auto gv = TensorAccessor(gv_args, gv_address);

    Noc noc_obj;
    CircularBuffer cb_output(cb_output_id);

    OpSignaler op_signaler;
    uint32_t self_write_done_sem_id = 0;
    if constexpr (fuse_op) {
        self_write_done_sem_id = get_arg_val<uint32_t>(arg_idx++);
        op_signaler = OpSignaler(arg_idx);
    }

    const uint32_t total = (row_end - row_start) * tiles_per_row;

    // 1. the local slice, from the local K/V tensors
    {
        SeqCursor cur;
        cur.r = row_start;
        for (uint32_t done = 0; done < total;) {
            const uint32_t n = std::min(total - done, num_tiles_to_write_per_packet);
            cb_output.reserve_back(num_tiles_to_write_per_packet);
            size_t l1_write_addr = cb_output.get_write_ptr();
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t page = cur.local_page();
                noc_async_read(cur.t == 0 ? k.get_noc_addr(page) : v.get_noc_addr(page), l1_write_addr, page_size);
                l1_write_addr += page_size;
                cur.advance();
            }
            noc_obj.async_read_barrier();
            cb_output.push_back(num_tiles_to_write_per_packet);
            done += n;
        }
    }

    // 2. the slices arriving from the other neighbour: wait per packet group, re-read for forwarding while due
    const uint32_t slices_expected = direction == 1 ? num_targets_backward_direction : num_targets_forward_direction;
    const uint32_t writes_expected = slices_expected - 1;  // the last received slice is not forwarded
    uint32_t sem_target = 0;
    for (uint32_t slices_received = 0; slices_received < slices_expected; ++slices_received) {
        uint32_t chip;
        if (direction == 1) {
            chip = my_chip_id + slices_received + 1;
            chip = chip >= ring_size ? chip - ring_size : chip;
        } else {
            chip = my_chip_id + ring_size - (slices_received + 1);
            chip = chip >= ring_size ? chip - ring_size : chip;
        }
        const bool should_forward = slices_received < writes_expected;

        // Signal the fused op as soon as the WHOLE slice has landed (every packet group's count), before the
        // re-read for forwarding: the consumer's per-shard gate must not wait behind the relay, and the relay
        // cannot start earlier anyway (the writer is still busy with the previous slice on the link).
        {
            const uint32_t n_packets = (total + num_tiles_to_write_per_packet - 1) / num_tiles_to_write_per_packet;
            const uint32_t n_syncs = (n_packets + chunks_per_sync - 1) / chunks_per_sync;
            noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + n_syncs);
        }
        if constexpr (fuse_op) {
            if (direction == 1 && slices_received == 0) {
                // the direction-1 writer pre-signals the local slice; do not overtake it
                Semaphore<> self_write_done_sem(self_write_done_sem_id);
                self_write_done_sem.wait_min(1);
                self_write_done_sem.set(0);
            }
            op_signaler.synchronize_workers_and_signal_op(chip);
        }

        SeqCursor cur;
        cur.r = row_start;
        uint32_t chunk_count = 0;
        for (uint32_t done = 0; done < total;) {
            const uint32_t n = std::min(total - done, num_tiles_to_write_per_packet);
            if (chunk_count % chunks_per_sync == 0) {
                noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), sem_target + 1);
                sem_target++;
            }
            chunk_count++;
            if (should_forward) {
                cb_output.reserve_back(num_tiles_to_write_per_packet);
                size_t l1_write_addr = cb_output.get_write_ptr();
                for (uint32_t j = 0; j < n; ++j) {
                    const uint32_t page = cur.gathered_page(chip);
                    noc_async_read(
                        cur.t == 0 ? gk.get_noc_addr(page) : gv.get_noc_addr(page), l1_write_addr, page_size);
                    l1_write_addr += page_size;
                    cur.advance();
                }
                noc_obj.async_read_barrier();
                cb_output.push_back(num_tiles_to_write_per_packet);
            } else {
                for (uint32_t j = 0; j < n; ++j) {
                    cur.advance();
                }
            }
            done += n;
        }
    }

    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem), 0);
    noc_obj.async_write_barrier();
    noc_obj.async_atomic_barrier();
}

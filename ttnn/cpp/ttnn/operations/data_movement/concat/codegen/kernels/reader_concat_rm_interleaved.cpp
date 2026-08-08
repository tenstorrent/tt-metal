// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Concat reader for RM interleaved tensors (non-width dims).
// Reads sticks from N_TENSORS input tensors in block order.
// Same block-cycling logic as tile reader, but uses noc_async_read with page_size
// instead of noc_async_read_tile.
//
// CT args layout:
//   [0]   cb_in
//   [1]   BATCH (read batch size)
//   [2]   N_TENSORS (=2)
//   [3]   ppb_0 (sticks per block for tensor 0)
//   [4]   ppb_1 (sticks per block for tensor 1)
//   [5]   CB_PAGE_SIZE (largest source/destination page)
//   [6]   IN0_PAGE_SIZE (source-0 physical page pitch)
//   [7]   IN1_PAGE_SIZE (source-1 physical page pitch)
//   [8..] TensorAccessorArgs for tensor 0, then tensor 1
//
// RT args layout:
//   [0]   num_sticks (total sticks this core reads)
//   [1]   start_tensor (which tensor to start reading from)
//   [2]   start_tensor_id (stick offset within that tensor's current block)
//   [3]   src_addr_0
//   [4]   src_addr_1
//   [5]   stick_id_0 (start stick id for tensor 0)
//   [6]   stick_id_1 (start stick id for tensor 1)
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    uint32_t num_sticks = get_arg_val<uint32_t>(0);
    uint32_t curr_tensor = get_arg_val<uint32_t>(1);
    uint32_t curr_tensor_id = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_in = get_compile_time_arg_val(0);
    constexpr uint32_t BATCH = get_compile_time_arg_val(1);
    constexpr uint32_t N_TENSORS = get_compile_time_arg_val(2);
    constexpr uint32_t ppb_0 = get_compile_time_arg_val(3);
    constexpr uint32_t ppb_1 = get_compile_time_arg_val(4);
    constexpr uint32_t CB_PAGE_SIZE = get_compile_time_arg_val(5);
    constexpr uint32_t IN0_PAGE_SIZE = get_compile_time_arg_val(6);
    constexpr uint32_t IN1_PAGE_SIZE = get_compile_time_arg_val(7);

    // TensorAccessor args start after the three page sizes.
    constexpr uint32_t ta_base = 8;
    constexpr auto ta0_args = TensorAccessorArgs<ta_base>();
    constexpr auto ta1_args = TensorAccessorArgs<ta0_args.next_compile_time_args_offset()>();

    // Runtime: src addresses and stick IDs
    const uint32_t src_addr_0 = get_arg_val<uint32_t>(3);
    const uint32_t src_addr_1 = get_arg_val<uint32_t>(4);

    uint32_t stick_id_0 = get_arg_val<uint32_t>(5);
    uint32_t stick_id_1 = get_arg_val<uint32_t>(6);

    // DRAM and interleaved L1 have different physical page alignment on BH.
    // Each accessor must use its source buffer's real pitch; using DRAM's 64B
    // pitch for a 16B-aligned L1 page silently addresses the wrong sticks.
    const auto s0 = TensorAccessor(ta0_args, src_addr_0, IN0_PAGE_SIZE);
    const auto s1 = TensorAccessor(ta1_args, src_addr_1, IN1_PAGE_SIZE);

    Noc noc;
    CircularBuffer input_cb(cb_in);

    uint32_t sticks_left = num_sticks;

    while (sticks_left > 0) {
        uint32_t batch = (sticks_left < BATCH) ? sticks_left : BATCH;
        input_cb.reserve_back(batch);
        uint32_t l1_offset = 0;

        for (uint32_t t = 0; t < batch; t++) {
            if (curr_tensor == 0) {
                noc.async_read(s0, input_cb, IN0_PAGE_SIZE, {.page_id = stick_id_0}, {.offset_bytes = l1_offset});
                stick_id_0++;
            } else {
                noc.async_read(s1, input_cb, IN1_PAGE_SIZE, {.page_id = stick_id_1}, {.offset_bytes = l1_offset});
                stick_id_1++;
            }
            l1_offset += CB_PAGE_SIZE;

            // Advance block tracking
            curr_tensor_id++;
            if (curr_tensor == 0 && curr_tensor_id == ppb_0) {
                curr_tensor_id = 0;
                curr_tensor = 1;
            } else if (curr_tensor == 1 && curr_tensor_id == ppb_1) {
                curr_tensor_id = 0;
                curr_tensor = 0;
            }
        }
        noc.async_read_barrier();
        input_cb.push_back(batch);
        sticks_left -= batch;
    }
}

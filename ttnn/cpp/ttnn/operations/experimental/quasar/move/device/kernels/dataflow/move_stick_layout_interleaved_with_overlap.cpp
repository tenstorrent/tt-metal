// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_pages = get_arg(args::num_pages);
    uint32_t controller_noc_x = get_arg(args::controller_noc_x);
    uint32_t controller_noc_y = get_arg(args::controller_noc_y);
    uint32_t control_value = get_arg(args::control_value);
    bool is_controller = get_arg(args::is_controller) == 1;
    uint32_t range_0_start_noc_x = get_arg(args::range_0_start_noc_x);
    uint32_t range_0_start_noc_y = get_arg(args::range_0_start_noc_y);
    uint32_t range_0_end_noc_x = get_arg(args::range_0_end_noc_x);
    uint32_t range_0_end_noc_y = get_arg(args::range_0_end_noc_y);
    uint32_t range_0_size = get_arg(args::range_0_size);
    uint32_t range_1_start_noc_x = get_arg(args::range_1_start_noc_x);
    uint32_t range_1_start_noc_y = get_arg(args::range_1_start_noc_y);
    uint32_t range_1_end_noc_x = get_arg(args::range_1_end_noc_x);
    uint32_t range_1_end_noc_y = get_arg(args::range_1_end_noc_y);
    uint32_t range_1_size = get_arg(args::range_1_size);
    uint32_t range_2_start_noc_x = get_arg(args::range_2_start_noc_x);
    uint32_t range_2_start_noc_y = get_arg(args::range_2_start_noc_y);
    uint32_t range_2_end_noc_x = get_arg(args::range_2_end_noc_x);
    uint32_t range_2_end_noc_y = get_arg(args::range_2_end_noc_y);
    uint32_t range_2_size = get_arg(args::range_2_size);
    uint32_t num_multicast_regions = get_arg(args::num_multicast_regions);
    uint32_t aligned_page_size = get_arg(args::aligned_page_size);

    constexpr uint32_t page_size = get_arg(args::page_size);

    Noc noc;
    DataflowBuffer cb(dfb::scratch);

    const auto src_addrgen = TensorAccessor(tensor::input);

    // if controller core then this local address will be incremented by remote cores,
    // otherwise controller core will set this to signal that write to dst can be done once controller core sees
    // control_value locally
    Semaphore sem(sem::sem);

    // read a ublock of tiles from src to CB
    cb.reserve_back(num_pages);
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        noc.async_read(
            src_addrgen,
            cb,
            page_size,
            {.page_id = i, .offset_bytes = 0},
            {.offset_bytes = (i - start_id) * aligned_page_size});
        noc.async_read_barrier();
    }

    if (is_controller) {
        sem.wait(control_value);

        // Signal the writers. Narrow grids can have fewer than 3 regions, so skip the empty ones.
        if (num_multicast_regions > 0) {
            sem.set_multicast<NocOptions::DEFAULT>(
                noc, range_0_start_noc_x, range_0_start_noc_y, range_0_end_noc_x, range_0_end_noc_y, range_0_size);
        }
        if (num_multicast_regions > 1) {
            sem.set_multicast<NocOptions::DEFAULT>(
                noc, range_1_start_noc_x, range_1_start_noc_y, range_1_end_noc_x, range_1_end_noc_y, range_1_size);
        }
        if (num_multicast_regions > 2) {
            sem.set_multicast<NocOptions::DEFAULT>(
                noc, range_2_start_noc_x, range_2_start_noc_y, range_2_end_noc_x, range_2_end_noc_y, range_2_size);
        }
    } else {
        // increment controller core semaphore
        sem.up(noc, controller_noc_x, controller_noc_y, 1);
        // wait for controller to signal write
        sem.wait(control_value);
    }

    // Publish the staged data to the writer kernel only AFTER the read barriers and the all-cores
    // handshake above, so the writer's wait_front(dfb::scratch) cannot drain CB -> dst until every
    // core has finished reading src (the overlap-safety invariant). Then drain the handshake's NoC
    // writes/atomics before returning (the Metal 2.0 FW epilogue does not).
    cb.push_back(num_pages);
    noc.async_full_barrier();
}

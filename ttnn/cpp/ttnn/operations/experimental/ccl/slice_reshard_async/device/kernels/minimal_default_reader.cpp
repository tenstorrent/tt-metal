// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/buffer_types.hpp"
#include "experimental/kernel_args.h"
#include "ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

constexpr bool is_first_chip = get_arg(args::is_first_chip);
constexpr bool is_last_chip = get_arg(args::is_last_chip);
constexpr bool direction = get_arg(args::direction);
const uint32_t stick_size = get_arg(args::stick_size);

template <uint32_t stick_size_bytes>
inline void zeroPad(const Noc& noc, uint32_t dfb_output_id) {
    // Zero-fill the DFB's current write entry via the device-side zero API. Self-contained:
    // it waits for the zero (write_zeros_l1_barrier), so the caller needs no separate barrier.
    DataflowBuffer dfb(dfb_output_id);
    noc.async_write_zeros(dfb, stick_size_bytes);
    noc.write_zeros_l1_barrier();
}

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Load the input tensor spec
    const uint32_t stick_start_id = get_arg(args::stick_start_id);
    const uint32_t num_sticks_to_read = get_arg(args::num_sticks_to_read);
    const uint32_t input_outer_dim_size = get_arg(args::input_outer_dim_size);
    const uint32_t outer_dims_to_forward = get_arg(args::outer_dims_to_forward);
    const uint32_t outer_dims_from_forward = get_arg(args::outer_dims_from_forward);
    const uint32_t outer_dims_to_keep_start = get_arg(args::outer_dims_to_keep_start);
    const uint32_t outer_dims_to_keep_end = get_arg(args::outer_dims_to_keep_end);
    const uint32_t num_sticks_per_outer_dim = get_arg(args::num_sticks_per_outer_dim);
    experimental::GlobalSemaphore out_ready_sem(global_sem::out_ready_sem);

    uint32_t read_size = stick_size;
    const auto src_accessor = TensorAccessor(tensor::src);

    Noc noc_obj;
    DataflowBuffer dfb_output(dfb::output);

    if (!is_last_chip) {
        // Read the "end" of each slice into the DFB to write to the neighbor
        for (uint32_t outer_dim_id = outer_dims_to_forward; outer_dim_id > 0; outer_dim_id--) {
            uint32_t src_stick_id = 0;
            if (direction) {
                src_stick_id = (input_outer_dim_size - outer_dim_id) * num_sticks_per_outer_dim + stick_start_id;
            } else {
                src_stick_id = (outer_dims_to_forward - outer_dim_id) * num_sticks_per_outer_dim + stick_start_id;
            }
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                dfb_output.reserve_back(1);
                noc_obj.async_read(src_accessor, dfb_output, read_size, {.page_id = src_stick_id}, {});

                src_stick_id++;

                noc_obj.async_read_barrier();
                dfb_output.push_back(1);
            }
        }
    } else {
        // If we need extend beyond the original input tensor, pad
        if (direction) {
            if (outer_dims_from_forward) {
                dfb_output.reserve_back(1);
                zeroPad<stick_size>(noc_obj, dfb::output);
                dfb_output.push_back(1);
            }
        }
    }

    if (direction) {
        for (uint32_t outer_dim_id = outer_dims_to_keep_start; outer_dim_id <= outer_dims_to_keep_end; outer_dim_id++) {
            uint32_t src_stick_id = outer_dim_id * num_sticks_per_outer_dim + stick_start_id;
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                dfb_output.reserve_back(1);
                noc_obj.async_read(src_accessor, dfb_output, read_size, {.page_id = src_stick_id}, {});

                src_stick_id++;

                noc_obj.async_read_barrier();
                dfb_output.push_back(1);
            }
        }
    }

    // Check that the semaphore is received
    if (!is_first_chip) {
        out_ready_sem.wait_min(1);
    }

    out_ready_sem.set(0);
}

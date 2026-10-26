// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct EntryCorrectionParams {
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    // The same number of cores per head, head by head: core j of a head takes its j-th share of the chunks and forms
    // every cores_per_head-th tile row of the final state from row j.
    std::vector<tt::tt_metal::CoreCoord> cores;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    // With a tail landing: head h's state lands on tail_landing_cores[h] (the relay's head cores, in head order), and
    // its arrival counter is the global semaphore at tail_landing_semaphore_address on that core.
    std::vector<tt::tt_metal::CoreCoord> tail_landing_cores;
    uint32_t tail_landing_semaphore_address = 0;
};

struct EntryCorrectionInputs {
    // Per chunk, the output for a zero seed F(0) and an identity seed F(I), height-sharded over the cores so each
    // core's chunk rows are in its own L1.
    Tensor zero_output;
    Tensor identity_output;
    // final = A @ entry + B.
    Tensor final_a;
    Tensor final_b;
    // The group's entry states, and the split tail's.
    Tensor entry;
    Tensor tail_entry;
    Tensor actual_start;
    std::optional<Tensor> actual_end;
    // The split tail's entry states as relay_affine_prefix(leave_final_in_landing=True) leaves them on the first
    // chronological rank: one K x K state per head core, stored by tile column, possibly still landing. Taken instead
    // of tail_entry; the first rank waits for every column and consumes the arrival count, every call.
    std::optional<Tensor> tail_landing;
};

}  // namespace ttnn::experimental::prim

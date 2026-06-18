// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/core_coord.hpp>

#include <cstdint>
#include <vector>

namespace ttnn::prim {

// Per-core runtime args for one ConcatS2I dispatch, derived purely from (tensor_args, output).
// SINGLE SOURCE OF TRUTH for both create_descriptor() (cache miss) and
// ConcatDeviceOperation::get_dynamic_runtime_args() (cache-hit re-apply).
//
// The only raw-address slot in this factory is the writer's output address (writer arg 0); the input
// addresses are CB `.buffer`-bound, not runtime args. Its index is recorded in writer_addr_indices so
// get_dynamic_runtime_args() can re-apply the live output address on every core on a cache hit (the
// factory bakes a plain address that would otherwise stay frozen at the first call's value).
struct ConcatS2IPerCoreArgs {
    std::vector<CoreCoord> cores;
    // Indexed by position in `cores`. Every core runs the S2I kernels, so every entry is populated.
    std::vector<std::vector<uint32_t>> reader_args;
    std::vector<std::vector<uint32_t>> writer_args;
    // Buffer-address arg positions in writer_args (same for every core); their value is the output address.
    std::vector<uint32_t> writer_addr_indices;
};

// Derive the S2I per-core core ordering and reader/writer runtime-arg vectors. Consumed by both
// create_descriptor() and get_dynamic_runtime_args().
ConcatS2IPerCoreArgs compute_concat_s2i_per_core_args(const ConcatInputs& tensor_args, const Tensor& output);

struct ConcatS2IProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatParams& operation_attributes, const ConcatInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim

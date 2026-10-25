// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include "sparse_push_plan_device_operation.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

// Hot rows of the sparse KV push. Runs on every requester chip after the all-gather of the sparse_push_plan tables.
// A row is hot when at least `threshold` chips need it: its owner multicasts it once to every chip instead of
// pushing it to each requester. Every chip derives the same hot set from the same tables, so the hot rows get the
// same slots everywhere, in a hot region at the front of every compact buffer: owner-major, and within an owner its
// rows of the prefix part (blocks before the chunk start's block) then those of the current part, each sub-run in
// local-row order, holding at most prefix_cap (current_cap) rows, padded to whole bank-lane units like the owner runs
// behind it. The op
//   - returns the hot table (same page layout as the plan table): [0] magic, [1] H = hot region slots, [2] threshold,
//     [3] active blocks, [4] split block, [32 + 2 * owner + part] sub-run base, [128, 128 + W) the hot rows
//     (within their caps) as a bitmap; and
//   - rewrites the plan's compact indices in place: a multicast row's index to its hot slot; any other (cold) row's to
//     its slot in the requester's run of rows from its owner, H slots up. The runs keep the plan's layout (sized by
//     every needed row) but now hold the cold rows only, packed: those of the current part first, then those of the
//     prefix part, each in local-row order, so an owner's push stays in whole packets.
struct SparsePushHotLayout {
    static constexpr uint32_t kMagic = 0x484F5431u;
    static constexpr uint32_t kBasesOff = SparsePushPlanLayout::kOwnerOff;
    static constexpr uint32_t kBitmapOff = SparsePushPlanLayout::kBitmapOff;
};

struct SparsePushHotParams {
    uint32_t num_devices = 0;
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t pages_per_packet = 0;
    uint32_t banks = 0;
    uint32_t threshold = 0;
    uint32_t prefix_cap = 0;
    uint32_t current_cap = 0;
};

struct SparsePushHotInputs {
    Tensor indices;      // uint32 row-major [1, 1, rows, topk] per device, the top-k rows the plan was built from
    Tensor compact;      // the plan's compact indices, rewritten in place
    Tensor table;        // this chip's plan table (its own needed rows and run layout)
    Tensor tables;       // the all-gathered plan tables, uint32 [1, 1, pages * num_devices, 1024]
    Tensor chunk_start;  // 1-element uint32: rows before it are the prefix part
};

struct SparsePushHotDeviceOperation {
    using operation_attributes_t = SparsePushHotParams;
    using tensor_args_t = SparsePushHotInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;  // the hot table

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

Tensor sparse_push_hot(
    const Tensor& indices,
    const Tensor& compact,
    const Tensor& table,
    const Tensor& tables,
    const Tensor& chunk_start,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks,
    uint32_t threshold,
    uint32_t prefix_cap,
    uint32_t current_cap);

}  // namespace ttnn::prim

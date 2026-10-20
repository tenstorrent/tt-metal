// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::high_bw_all_gather {

// PROTOTYPE on-device planner for the multicast-tree push (high_bw_all_gather with SPARSE_PUSH_TREE=2). From the
// all-gathered sparse_push_plan tables of every requester, each chip builds the record streams of its 16 senders
// (see kernels/tree_push_reader2.cpp) and rewrites its own top-k indices to slots of its tree-push buffer. Rules and
// layout match the host reference _tree_plan_records in
// tests/nightly/blackhole/sdpa/test_sparse_sdpa_gather_overlap.py.
struct SparsePushTreePlanParams {
    uint32_t mesh_rows = 0;
    uint32_t mesh_cols = 0;
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t pages_per_packet = 0;
    uint32_t banks = 0;
    uint32_t links = 2;
    uint32_t senders_per_link = 2;
    uint32_t levels = 12;
    uint32_t stage_rows = 65536;
    uint32_t stream_records = 4096;  // records per sender page
    uint32_t slot_capacity = 65536;  // rows a requester may need
};

struct SparsePushTreePlanInputs {
    Tensor tables;   // uint32 [1, 1, pages * num_devices, 1024]: every requester's sparse_push_plan table
    Tensor indices;  // uint32 [1, 1, rows, topk] per device, 0xFFFFFFFF = masked
    Tensor rank;     // uint32 [1, 1, 1, 16] per device: word 0 = this device's row-major mesh rank
};

struct SparsePushTreePlanDeviceOperation {
    using operation_attributes_t = SparsePushTreePlanParams;
    using tensor_args_t = SparsePushTreePlanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;  // {sender streams, compact indices, slot table}

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& attrs, const tensor_args_t& args, tensor_return_value_t& outputs);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::high_bw_all_gather

namespace ttnn::prim {

std::vector<Tensor> sparse_push_tree_plan(
    const Tensor& tables,
    const Tensor& indices,
    const Tensor& rank,
    const ttnn::operations::experimental::high_bw_all_gather::SparsePushTreePlanParams& params);

}  // namespace ttnn::prim

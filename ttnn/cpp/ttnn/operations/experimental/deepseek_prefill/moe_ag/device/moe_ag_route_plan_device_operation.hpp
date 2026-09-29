// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

#include "moe_ag_route_plan_device_operation_types.hpp"

namespace ttnn::prim {

// All-gather MoE route plan: from the dispatch group's gathered top-k [T, K] and this chip's local-slot map, on device
//   counts [1, NG] / regions [1, NG]: tokens per global expert and each local expert's first flat row (local order,
//     32-row aligned; 0 for other chips' experts)
//   token_index [1, rows]: flat row -> gathered token (the tile tail of each region zero)
//   y_slot [1, T K]: per (token, k) the flat row of its expert output, 0xFFFFFFFF if not a local expert
// One program on an 8 x 8 core rectangle (range histograms, prefix scans and the lists exchanged over the NoC).
struct MoeAgRoutePlanDeviceOperation {
    using operation_attributes_t = MoeAgRoutePlanParams;
    using tensor_args_t = MoeAgRoutePlanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;

    struct ProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ProgramFactory>;

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

// Cores of the plan (an 8 x 8 rectangle: the histogram table is multicast over it) and its limits.
inline constexpr uint32_t MOE_AG_ROUTE_PLAN_CORES = 64;
inline constexpr uint32_t MOE_AG_ROUTE_PLAN_MAX_K = 32;  // top-k ids staged in 64 B L1 pages

std::vector<Tensor> moe_ag_route_plan(
    const Tensor& topk_indices,
    const Tensor& local_slot_map,
    uint32_t experts_per_chip,
    uint32_t num_rows,
    const std::vector<Tensor>& preallocated_outputs = {});

}  // namespace ttnn::prim

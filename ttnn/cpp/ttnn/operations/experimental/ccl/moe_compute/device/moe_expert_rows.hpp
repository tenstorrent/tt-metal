// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The expert rows path of moe_compute, two programs. The expert rows program computes one unweighted bf16 row per
// routed (token, local expert) into a DRAM row buffer and writes the routing table; the placement program
// (MoEComputePlaceFactory, a program factory of MoEComputeDeviceOperation) turns them into moe_compute's outputs. Each
// expert's weights are streamed once per job of at most 32 * row_tiles of its rows, split over expert groups of
// readers_per_bank reader cores per DRAM bank, read from moe_compute's prepared weight layout as it is. With one row
// tile per job, x is one whole-hidden slot per job; with several (weight-stationary jobs), x moves in K-chunks and FP32
// partials stay in L1, so the rows stay bitwise equal to one row tile per job.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>

#include "moe_compute_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Shape of one expert rows call, from the call's tensors and attributes.
struct MoEExpertRowsShape {
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t local_experts = 0;
    uint32_t top_k = 0;
    uint32_t tokens = 0;
    uint32_t global_experts = 0;
    uint32_t num_sources = 0;
    uint32_t index_page_bytes = 0;    // aligned page of the indices / scores tensors (one token's top-k)
    uint32_t mapping_page_bytes = 0;  // aligned page of the expert mapping tensor (one device's row)
    bool has_bias = false;

    uint32_t hidden_tiles() const { return hidden_size / tt::constants::TILE_WIDTH; }
    uint32_t intermediate_tiles() const { return intermediate_size / tt::constants::TILE_WIDTH; }
    uint32_t bias_rows() const { return has_bias ? 1 : 0; }  // the prepared weights' bias K row
};

struct MoEExpertRowsParams {
    MoEExpertRowsShape shape;
    uint32_t layer_id = 0;
    uint32_t cluster_axis = 1;  // dispatch axis: the sources of the tokens (moe_compute's tilize rule)
    uint32_t cb_budget = 0;     // bucketed CB bytes per core the plan may use (program-cache key)
    uint32_t row_tiles = 1;     // row tiles per job
    ttnn::experimental::prim::detail::MoEActivationFunction activation_type =
        ttnn::experimental::prim::detail::MoEActivationFunction::SILU;
    std::optional<float> activation_limit;
    tt::tt_metal::MathFidelity math_fidelity = tt::tt_metal::MathFidelity::LoFi;
    bool fp32_dest_acc_en = false;

    auto attributes() const {
        using ttsl::reflection::Attribute;
        std::vector<std::tuple<std::string, Attribute>> attrs;
        attrs.reserve(18);
        attrs.emplace_back("hidden_size", shape.hidden_size);
        attrs.emplace_back("intermediate_size", shape.intermediate_size);
        attrs.emplace_back("local_experts", shape.local_experts);
        attrs.emplace_back("top_k", shape.top_k);
        attrs.emplace_back("tokens", shape.tokens);
        attrs.emplace_back("global_experts", shape.global_experts);
        attrs.emplace_back("num_sources", shape.num_sources);
        attrs.emplace_back("index_page_bytes", shape.index_page_bytes);
        attrs.emplace_back("mapping_page_bytes", shape.mapping_page_bytes);
        attrs.emplace_back("has_bias", shape.has_bias);
        attrs.emplace_back("layer_id", layer_id);
        attrs.emplace_back("cluster_axis", cluster_axis);
        attrs.emplace_back("cb_budget", cb_budget);
        attrs.emplace_back("row_tiles", row_tiles);
        attrs.emplace_back("activation_type", static_cast<uint32_t>(activation_type));
        attrs.emplace_back("activation_limit", activation_limit);
        attrs.emplace_back("math_fidelity", static_cast<uint32_t>(math_fidelity));
        attrs.emplace_back("fp32_dest_acc_en", fp32_dest_acc_en);
        return attrs;
    }
};

struct MoEExpertRowsInputs {
    const ttnn::Tensor& input_tensor;
    const ttnn::Tensor& expert_indices_tensor;
    const ttnn::Tensor& expert_scores_tensor;
    const ttnn::Tensor& expert_mapping_tensor;
    const ttnn::Tensor& w0_w1_tensor;
    const ttnn::Tensor& w2_tensor;
};

struct MoEExpertRowsFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MoEExpertRowsParams& args,
        const MoEExpertRowsInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MoEExpertRowsParams& args,
        const MoEExpertRowsInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

struct MoEExpertRowsDeviceOperation {
    using operation_attributes_t = MoEExpertRowsParams;
    using tensor_args_t = MoEExpertRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<MoEExpertRowsFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// The placement program's inputs: moe_compute's, and the expert rows program's row buffer and routing table.
struct MoEComputePlaceInputs {
    const ttnn::Tensor& tilize_input_tensor;
    const ttnn::Tensor& tilize_expert_indices_tensor;
    const ttnn::Tensor& tilize_expert_scores_tensor;
    const ttnn::Tensor& tilize_expert_mapping_tensor;
    const ttnn::Tensor& matmul_w0_w1_tensor;
    const ttnn::Tensor& matmul_w2_tensor;
    const std::optional<ttnn::Tensor>& optional_output_tensor;
    const ttnn::Tensor& row_buffer_tensor;
    const ttnn::Tensor& routing_table_tensor;

    MoEComputeInputs moe_compute_inputs() const {
        return {
            tilize_input_tensor,
            tilize_expert_indices_tensor,
            tilize_expert_scores_tensor,
            tilize_expert_mapping_tensor,
            matmul_w0_w1_tensor,
            matmul_w2_tensor,
            optional_output_tensor};
    }
};

// The placement program of an expert rows call: moe_compute's outputs from the row buffer and the routing table, and
// with a combine (SingleDevice, FullCcl) every routed row sent to the combine output.
struct MoEComputePlaceFactory {
    static tt::tt_metal::WorkloadDescriptor create_workload_descriptor(
        const MoEComputeParams& args,
        const MoEComputePlaceInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords);

    static void override_runtime_arguments(
        tt::tt_metal::Program& program,
        const MoEComputeParams& args,
        const MoEComputePlaceInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<ttnn::MeshCoordinate>& coord = std::nullopt);
};

// moe_compute's outputs, specs and checks; one placement program serves every layer and every compute setting (it
// reads neither the weights nor the activation, fidelity or DEST mode).
struct MoEComputePlaceDeviceOperation {
    using operation_attributes_t = MoEComputeParams;
    using tensor_args_t = MoEComputePlaceInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<MoEComputePlaceFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

// The expert rows parameters when this call should run the expert rows programs instead of the ring program, else
// nullopt. Logs the choice and its reason at debug level.
std::optional<MoEExpertRowsParams> select_moe_compute_expert_rows(
    const MoEComputeParams& args, const MoEComputeInputs& tensor_args);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// The expert rows program: [rows, routing table]. rows: bf16 ROW_MAJOR DRAM
// [min(tokens * top_k, local_experts * tokens), hidden], row r = the r-th routed (token, k) of a local expert in
// (expert ascending, token ascending) order, at most `tokens` per expert (its e_t page; a token that names the expert
// at several k is listed once per k, and listings past the page are dropped); rows past the routed count are not
// written.
std::vector<ttnn::Tensor> moe_expert_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::experimental::prim::MoEExpertRowsParams& params);

}  // namespace ttnn::prim

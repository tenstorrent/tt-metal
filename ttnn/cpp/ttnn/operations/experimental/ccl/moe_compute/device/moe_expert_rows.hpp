// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// expert rows for moe_compute: program 1 computes one unweighted bf16 row per routed (token, local expert)
// into a DRAM row buffer and writes the routing table; program 2 (MoEComputePlaceFactory, a program factory of
// MoEComputeDeviceOperation) turns them into moe_compute's outputs. Each expert's weights are streamed once per job
// of at most 32 M of its rows (M row tiles), split over G expert groups of k reader cores per DRAM bank, read from
// moe_compute's prepared weight layout as it is. M = 1 keeps a whole-hidden x slot per job; M > 1 (weight-stationary)
// moves x in K-chunks and keeps FP32 partials in L1, so the rows stay bitwise equal to M = 1.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/base_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>

#include "moe_compute_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

// Byte offsets of the routing scratch page of the expert-row reader (cb_rt). The table block [table, jobs) is what
// the root core writes to DRAM for program 2.
struct MoEExpertRowsRoutingLayout {
    uint32_t own = 0, maps = 0, ids = 0, scores = 0, ctl = 0, slots = 0, counts = 0, offsets = 0;
    uint32_t table = 0, table_counts = 0, table_offsets = 0, rows = 0;
    uint32_t entry_slots = 0, entry_tokens = 0, entry_scores = 0, entry_k = 0;
    uint32_t jobs = 0, areas = 0, size = 0;
    uint32_t table_bytes() const { return jobs - table; }
};

// One core of the expert-row program.
struct MoEExpertRowsCore {
    CoreCoord core;
    uint32_t ring_pos = 0;      // shard index of the prepared weights = position in moe_compute's ring
    uint32_t bank = 0;          // DRAM bank holding that shard
    uint32_t group = 0;         // expert group (jobs dealt by rows_reader.cpp)
    uint32_t g0 = 0, ng = 0;    // W0/W1 two-column groups of the shard
    bool half_col = false;      // the last of them is the shard's half block-column (one column, 2 tiles per K row)
    uint32_t c0 = 0, na = 0;    // real intermediate columns
    uint32_t q0 = 0, nq = 0;    // W2 four-tile output groups of the shard
    bool half_out = false;      // the last of them is the half-width last a2a iteration (2 tiles per K row)
    uint32_t n0 = 0, nout = 0;  // real output tiles
    uint32_t reader_noc = 0;
};

struct MoEExpertRowsPlan {
    uint32_t row_tiles = 1;  // M: row tiles per job (one weight pass)
    uint32_t readers_per_bank = 0;
    uint32_t groups = 0;
    uint32_t cores_per_group = 0;
    uint32_t x_slots = 0;
    uint32_t a2_slots = 0;
    uint32_t a_tiles = 0;        // cb_a pages: two per W0/W1 group of the busiest core
    uint32_t block_tiles = 0;    // weight tiles per read block
    uint32_t block_packets = 0;  // NoC packets per read block
    uint32_t block_slots = 0;
    uint32_t blocks_in_flight = 0;
    uint32_t sources_per_core = 0;
    // hidden tiles per x chunk (all of them with one row tile per job), the most one core owns of a chunk, weight
    // blocks per chunk unit (the weight tiles of one chunk of one column group, or of one W2 chunk of one output
    // group), W2 rows per chunk
    uint32_t chunk_tiles = 0;
    uint32_t chunk_piece_max = 0;
    uint32_t chunk_blocks = 0;
    uint32_t w2_chunk_rows = 0;
    // slots handed back by credits (several chunks per job); without, the a exchange orders slot reuse (one chunk per
    // job and at least three a2 slots, rows_writer.cpp)
    bool slot_credits = false;
    uint32_t ctl_bytes = 0;
    uint32_t cb_bytes = 0;  // circular buffers per core
    MoEExpertRowsRoutingLayout routing;
    std::vector<MoEExpertRowsCore> cores;  // group-major
};

// Shape of one expert rows call, from the call's tensors and attributes.
struct MoEExpertRowsShape {
    uint32_t hidden_size = 0;
    uint32_t intermediate_size = 0;
    uint32_t local_experts = 0;
    uint32_t top_k = 0;
    uint32_t tokens = 0;
    uint32_t global_experts = 0;
    uint32_t num_sources = 0;
    uint32_t index_page_bytes = 0;  // aligned page of the indices / scores tensors (one token's top-k)
    uint32_t row_cap = 0;           // rows kept per local expert: its e_t page (tokens) and double-buffer half
    bool has_bias = false;
};

// The plan for `shape` with jobs of `row_tiles` row tiles on this device within `cb_budget` bytes of circular buffers
// per core, or why there is none. shard_banks[r]: DRAM bank of the prepared weights' shard r (= moe_compute ring
// position r).
std::optional<MoEExpertRowsPlan> plan_moe_expert_rows(
    ttnn::MeshDevice* mesh_device,
    const MoEExpertRowsShape& shape,
    const std::vector<uint32_t>& shard_banks,
    uint32_t cb_budget,
    uint32_t row_tiles,
    bool fp32_dest_acc_en,
    std::string& refusal);

struct MoEExpertRowsParams {
    MoEExpertRowsShape shape;
    uint32_t layer_id = 0;
    uint32_t cluster_axis = 1;  // dispatch axis: the sources of the tokens (moe_compute's tilize rule)
    uint32_t cb_budget = 0;     // bucketed CB bytes per core the plan may use (program-cache key)
    uint32_t row_tiles = 1;     // M: row tiles per job
    ttnn::experimental::prim::detail::MoEActivationFunction activation_type =
        ttnn::experimental::prim::detail::MoEActivationFunction::SILU;
    float activation_limit = 0.0f;
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
        attrs.emplace_back("row_cap", shape.row_cap);
        attrs.emplace_back("has_bias", shape.has_bias);
        // layer_id is not a key: it only offsets the weight addresses (runtime arguments)
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
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel = 0;
        tt::tt_metal::KernelHandle writer_kernel = 0;
        std::vector<CoreCoord> cores;
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MoEExpertRowsParams& args,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const MoEExpertRowsInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MoEExpertRowsParams& args,
        const MoEExpertRowsInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
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

// Program 2 of an expert rows call: moe_compute's outputs from the row buffer and the routing table (ComputeOnly), or
// the metadata, the double-buffer feed and selective_reduce_combine's kernels (FullLocal, FullCcl).
struct MoEComputePlaceFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle place_kernel = 0;
        std::vector<CoreCoord> cores;  // place.cpp's cores; with the combine also feed.cpp's, in ring order
        std::optional<tt::tt_metal::KernelHandle> feed_kernel;
        tt::tt_metal::KernelHandle combine_reader = 0;
        tt::tt_metal::KernelHandle combine_writer = 0;
        tt::tt_metal::CBHandle combine_data_cb = 0;
        std::vector<CoreCoord> combine_cores;
        std::vector<GlobalSemaphore> combine_semaphores;  // FullCcl: the combine's init and final barriers
    };
    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const MoEComputeParams& args,
        const ttnn::MeshCoordinateRangeSet& mesh_coordinates,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);

    static ttnn::device_operation::CachedProgram<shared_variables_t> create_at(
        const MoEComputeParams& args,
        const ttnn::MeshCoordinate& mesh_coordinate,
        const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value,
        const std::optional<GlobalSemaphore>& init_barrier_semaphore,
        const std::optional<GlobalSemaphore>& final_barrier_semaphore);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const MoEComputeParams& args,
        const MoEComputeInputs& tensor_args,
        std::vector<ttnn::Tensor>& tensor_return_value);
};

// The expert rows parameters when this call should run the expert rows programs instead of the ring program, else
// nullopt. Logs the choice and its reason at debug level.
std::optional<MoEExpertRowsParams> select_moe_compute_expert_rows(
    const MoEComputeParams& args, const MoEComputeInputs& tensor_args);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

// Program 1: [rows, routing table]. rows: bf16 ROW_MAJOR DRAM [min(tokens * top_k, local_experts * row_cap), hidden],
// row r = the r-th routed (token, k) of a local expert in (expert ascending, token ascending) order, at most row_cap
// per expert; rows past the routed count are not written.
std::vector<ttnn::Tensor> moe_expert_rows(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& expert_indices_tensor,
    const ttnn::Tensor& expert_scores_tensor,
    const ttnn::Tensor& expert_mapping_tensor,
    const ttnn::Tensor& w0_w1_tensor,
    const ttnn::Tensor& w2_tensor,
    const ttnn::experimental::prim::MoEExpertRowsParams& params);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// Where a weight's tiles sit in a chunked weight, for a grid num_columns cores wide. Each core column (share) takes
// share_n_tiles of an expert's N tiles (the last share last_share_n_tiles); its slice of an expert is
// (K + bias) x its width tiles in slot order (each output subblock's K rows of subblock_w tiles, then its bias
// tiles), cut into chunks of chunk_tiles. Chunk j of share s's slice of expert e is the contiguous run of tiles
//     bank * shard_tiles + ((e * max_chunks + j) * bank_sharers + s / 8) * chunk_tiles
// of a (1, 1, 32, 8 * shard_tiles * 32) TILE tensor WIDTH_SHARDED over the 8 DRAM banks, bank (s + j + e) % 8, so
// the shares read distinct banks at each step.
struct GroupedWeightChunks {
    uint32_t num_shares = 0;
    uint32_t share_n_tiles = 0;
    uint32_t last_share_n_tiles = 0;
    uint32_t subblock_w = 0;
    uint32_t chunk_tiles = 0;
    uint32_t max_chunks = 0;
    uint32_t bank_sharers = 0;
    uint32_t shard_tiles = 0;
};

inline constexpr uint32_t kGroupedWeightBanks = 8;

GroupedWeightChunks grouped_weight_chunks(
    uint32_t k_tiles, uint32_t n_tiles, uint32_t num_experts, bool has_bias, uint32_t chunk_tiles, uint32_t grid_x);

// Sizes of one call and the cores it runs on, derived from its inputs, attributes and device.
struct GroupedMatmulW1Layout {
    uint32_t num_experts = 0;
    uint32_t rows = 0;
    uint32_t k_tiles = 0;
    uint32_t n_tiles = 0;
    // Blocks are dealt round-robin to num_lanes core rows, or with per_expert_lanes lane e takes
    // expert e's blocks, num_lanes = num_experts. The N tiles split over num_columns core columns,
    // column_n_tiles each and last_column_n_tiles on the last.
    uint32_t num_lanes = 0;
    uint32_t num_columns = 0;
    uint32_t column_n_tiles = 0;
    uint32_t last_column_n_tiles = 0;
    // Output tiles a matmul_block call computes, at most what DST holds.
    uint32_t subblock_w = 0;
    bool has_bias = false;
    // Weight tiles a slot holds: (K + bias) x column_n_tiles; slots in the ring (1 with one expert).
    uint32_t slot_tiles = 0;
    uint32_t weight_slots = 0;
    // The chunked weight's geometry; chunk_tiles 0 for a plain TILE weight, read one tile at a time in slot order.
    GroupedWeightChunks chunks;
    // Weight reads in flight a share sender keeps.
    uint32_t weight_depth = 0;
    // Dedicated input senders a lane, east of the compute grid (0: the lane's east core sends), and the blocks of
    // read-ahead each keeps.
    uint32_t input_senders = 0;
    uint32_t staging_blocks = 0;
    // The folded layout: lanes 0 to grid rows - 1 are core rows over the share columns, each sent by row_lane_senders
    // cores east of the shares (or by its east core); the columns east of those are lanes too, share s at row s, each
    // sent by input_senders cores below it. A share's weights reach those columns through a forwarder in the share's
    // column.
    bool folded = false;
    // In the folded layout, the dedicated senders a row lane (0: its east core sends).
    uint32_t row_lane_senders = 0;
    // Input and output ring slots, in blocks.
    uint32_t input_slots = 0;
    uint32_t output_slots = 0;
    // One expert: compute runs subblock-outer over groups of group_blocks blocks (0: the lane's whole set) and the
    // call's tail blocks split into one-subblock units.
    bool subblock_outer = false;
    uint32_t group_blocks = 0;
    bool tail_split = false;
};

struct GroupedMatmulW1DeviceOperation {
    struct operation_attributes_t {
        tt::tt_metal::DataType dtype = tt::tt_metal::DataType::BFLOAT16;
        bool transpose_weight = false;
        std::optional<ttnn::operations::unary::UnaryWithParam> activation;
        tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE;
        tt::tt_metal::MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
        bool per_expert_lanes = false;
        // A chunked weight (prepare_grouped_matmul_w1_weight): its chunk size, an expert's N, and whether it holds the
        // bias. weight_chunk_tiles 0: a plain TILE weight.
        uint32_t weight_chunk_tiles = 0;
        uint32_t weight_n = 0;
        bool weight_has_bias = false;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& weight;
        // None: one expert, taking every row.
        std::optional<Tensor> expert_counts;
        std::optional<Tensor> bias;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Validates the call and derives its sizes. compute_output_specs calls it first because the
    // framework allocates the output before validate_on_program_cache_miss runs.
    static GroupedMatmulW1Layout compute_layout(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Every runtime argument that is not a buffer address comes from a hashed attribute or tensor
    // spec, so the default program hash keeps a cache hit correct; the framework re-binds the buffers.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor grouped_matmul_w1(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    tt::tt_metal::Layout output_layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool per_expert_lanes,
    const std::optional<Tensor>& bias,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    bool weight_has_bias);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

// The cores of one call, on the 11 x 10 grid the layout is built for. Shares are core columns 0 to 7, each
// multiplying by 3 of the expert's 24 N tiles. Row lanes are core rows (0..7, y); with column lanes, columns 8 to 10
// at rows 2 to 9 are lanes too, the core at row r holding share 7 at r = 2 and r - 3 above. Share s's top core (s, 0)
// streams its weights down its column; with column lanes a forwarder ((s, s + 3), (7, 1) for share 7) passes them
// east to its share's column-lane cores. A lane's input sender reads each block from the lane's own cores and
// multicasts it: (8, 0) and (8, 1), which do not compute, for row lanes 0 and 1; (7, y) for the other row lanes and
// (x, 9) for the column lanes.
struct GroupedMatmulW2Map {
    uint32_t num_lanes = 0;  // 10, or 13 with the column lanes
    uint32_t lane_cores = 0;
    uint32_t num_shares = 0;

    bool column_lanes() const { return num_lanes > kRowLanes; }
    CoreCoord lane_core(uint32_t lane, uint32_t index) const;
    CoreCoord input_sender(uint32_t lane) const;
    bool dedicated_sender(uint32_t lane) const { return lane < kDedicatedSenderLanes; }
    uint32_t share_of(const CoreCoord& core) const;
    CoreCoord forwarder(uint32_t share) const;
    uint32_t forward_row(uint32_t share) const;
    std::vector<CoreCoord> compute_cores() const;
    // The input's shard order: entry input_pieces * turn + piece is lane turn % lanes's core
    // input_pieces * (turn / lanes) + piece.
    std::vector<CoreCoord> input_shard_cores(uint32_t input_pieces) const;

    static constexpr uint32_t kRowLanes = 10;
    static constexpr uint32_t kDedicatedSenderLanes = 2;
};

// The map for an input of input_blocks 32-row blocks: column lanes from kColumnLaneBlocks blocks on.
GroupedMatmulW2Map grouped_matmul_w2_map(uint32_t input_blocks, const CoreCoord& grid);

inline constexpr uint32_t kGroupedMatmulW2InputPieces = 4;
inline constexpr uint32_t kGroupedMatmulW2InputSlots = 6;
inline constexpr uint32_t kGroupedMatmulW2OutputSlots = 3;

// The input layout grouped_matmul_w2 reads: (1, 1, rows, K) ND-sharded in L1 in shards of 32 x K / 4 (a quarter of a
// block), round-robin over the lanes' cores in input_shard_cores' order, so each block's pieces sit in its own lane.
tt::tt_metal::MemoryConfig grouped_matmul_w2_input_memory_config(uint32_t rows, uint32_t k, const CoreCoord& grid);

struct GroupedMatmulW2DeviceOperation {
    struct operation_attributes_t {
        tt::tt_metal::MemoryConfig memory_config;
        DeviceComputeKernelConfig compute_kernel_config;
        // The prepared weight's chunk size and an expert's N.
        uint32_t weight_chunk_tiles = 0;
        uint32_t weight_n = 0;
        // Half slices each weight ring holds (compute/grouped_matmul_w2.cpp).
        uint32_t first_slots = 2;
        uint32_t second_slots = 2;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& weight;
        const Tensor& expert_counts;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // Every runtime argument that is not a buffer address comes from a hashed attribute or tensor spec, so the default
    // program hash keeps a cache hit correct; the framework re-binds the buffers.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

// Circular-buffer bytes a core of the call takes, for the model's L1 planning.
uint32_t grouped_matmul_w2_circular_buffer_bytes(
    uint32_t k, uint32_t n, uint32_t first_slots, uint32_t second_slots, uint32_t num_experts, const CoreCoord& grid);

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor grouped_matmul_w2(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& expert_counts,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    uint32_t weight_chunk_tiles,
    uint32_t weight_n,
    uint32_t first_slots,
    uint32_t second_slots);

}  // namespace ttnn::prim

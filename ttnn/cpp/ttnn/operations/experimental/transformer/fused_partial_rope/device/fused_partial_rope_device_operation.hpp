// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tile.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::transformer::fused_partial_rope {

// ROW_MAJOR activations are consumed as 1x32 faces (one row). TILE X uses its native tile.
// A ROW_MAJOR spec cannot carry Tile(1, 32); compute still packs those faces, and for dense
// dtypes a 1x32 face is 32 contiguous row elements that write straight into the ROW_MAJOR
// buffer.
inline tt::tt_metal::Tile input_tile_for_compute(const Tensor& input) {
    if (input.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        return tt::tt_metal::Tile({1, tt::constants::TILE_WIDTH}, false);
    }
    return input.tensor_spec().tile();
}

// -----------------------------------------------------------------------------
// FusedPartialRopeDeviceOperation
//
// Fuses the deepseek_v4_flash `_apply_rope` calc into one device op: interleaved
// RoPE on each `head_dim`-wide block of a sharded `[1, 1, rows, D]` input
// (`D % head_dim == 0`; `head_dim == D` is the single-block case). Within a
// block the trailing `rope_dim` channels are rotated and the leading
// `head_dim - rope_dim` "nope" channels pass through:
//
//   out[..., b, :Hd-Rd] = x[..., b, :Hd-Rd]
//   out[..., b, Hd-Rd:] = x_rope * cos + (x_rope @ trans_mat) * sin
//
// Two input memory layouts are supported, each with its own program factory:
//   * height-sharded L1: TILE is one tile-row (32 rows) per core, so
//     num_cores = ceil(rows / 32). Core i owns input tile-row i. ROW_MAJOR is
//     many 1x32 faces: each core holds `shard_height` rows of the full D.
//   * width-sharded L1: every core holds all rows but only a `shard_width`
//     column slice of D. A core's columns can cover nope, rope, or both,
//     and with multiple head blocks they can straddle several such
//     boundaries. ROW_MAJOR uses 1-high faces, so Ht = shard height (one
//     face per input row).
// In both cases `cos`/`sin` are `[1, 1, rows, Rd]` (or a single broadcast row)
// DRAM-interleaved TILE tables streamed per-core by the reader (shared across
// every head block of a row), and `trans_mat` is a single [32, 32]
// rotate_half tile, replicated. ROW_MAJOR X requires the broadcast (one
// logical cos/sin row) and is computed as 1x32 faces. The rotation is
// block-diagonal per tile (it pairs channels 2p / 2p+1), so each rope tile
// rotates independently of how the columns are spread over cores. Output
// layout matches the input.
// -----------------------------------------------------------------------------
struct FusedPartialRopeDeviceOperation {
    struct operation_attributes_t {
        uint32_t rope_dim;
        uint32_t head_dim;  // resolved; never 0 (defaults to D at invoke)
        MemoryConfig output_mem_config;
        ttnn::DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const Tensor& cos;
        const Tensor& sin;
        const Tensor& trans_mat;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ShardedProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    struct WidthShardedProgramFactory {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<ShardedProgramFactory, WidthShardedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::fused_partial_rope

namespace ttnn::prim {

ttnn::Tensor fused_partial_rope(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos,
    const ttnn::Tensor& sin,
    const ttnn::Tensor& trans_mat,
    uint32_t rope_dim,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    uint32_t head_dim);

}  // namespace ttnn::prim

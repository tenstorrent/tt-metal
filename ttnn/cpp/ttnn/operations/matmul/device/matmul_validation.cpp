// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_validation.hpp"

#include <string_view>

#include <fmt/format.h>

#include "tt-metalium/experimental/global_circular_buffer.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "tt_stl/reflection.hpp"
#include "ttnn/global_circular_buffer.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

namespace {

using operations::matmul::utilities::get_M_dim;
using operations::matmul::utilities::get_N_dim;
using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;
using tt::tt_metal::TensorSpec;

// ===========================================================================
// VALIDATIONS FOR ALL CONFIGS that need the chosen program config.
// ===========================================================================

// Tiny Tile Constraints: runs after the program config is chosen. Rejects tiny-tile
// combos that hang or deadlock on specific paths. See #42927.
std::string validate_matmul_tiny_tile_constraints(
    const TensorSpec& input_tensor_b,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const bool uses_tiny_outer_tile = (in0_tile.get_height() != TILE_HEIGHT || in1_tile.get_width() != TILE_WIDTH);
    if (!uses_tiny_outer_tile) {
        return {};
    }

    // Transposed in1 with narrow tile width (16) is not supported by the LLK matmul
    // path: llk_math_matmul has no addr_mod handling for 32x16 transposed tiles.
    // Without this check the kernel hangs (CB producer/consumer deadlock). See #42927.
    const bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
    if (in1_transpose_tile && in1_tile.get_width() == 16) {
        return fmt::format(
            "matmul does not support transposed in1 with tile width 16 (in1 tile is {}x{} with transpose). "
            "Use tile width 32 for transposed in1, or disable transpose_tile.",
            in1_tile.get_height(),
            in1_tile.get_width());
    }

    // Bfp compressed in1 dtypes (BFLOAT8_B, BFLOAT4_B) on the 2D/1D mcast paths hang for
    // tile_h < 16 — the LLK unpack/pack path for Bfp faces is not yet validated below
    // face_height 16 on these factories. The MatmulMultiCoreReuseProgramConfig path does
    // support smaller tile_h with Bfp dtypes, so this check is scoped to the mcast configs.
    // See #42927. This is a "not currently supported" rejection, not a permanent rule — it
    // should be removed when the underlying kernel limitation is resolved.
    const bool in1_is_bfp =
        (input_tensor_b.data_type() == DataType::BFLOAT8_B) || (input_tensor_b.data_type() == DataType::BFLOAT4_B);
    const bool is_mcast_config =
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(chosen_program_config) ||
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
    if (in1_is_bfp && is_mcast_config && in0_tile.get_height() < 16) {
        return fmt::format(
            "matmul tiny-tile combo with in1 dtype {} and tile_h {} is not currently supported on the "
            "mcast program config path (requires tile_h >= 16); see issue #42927",
            input_tensor_b.data_type(),
            in0_tile.get_height());
    }
    return {};
}

// Batch Compatibility: checks bcast -> B must be single-batch; non-bcast -> A and B must
// match rank + batch dims, unless A's batch is 1 and reused across B (the Mcast1D
// in0-reuse exception).
std::string validate_matmul_batch_compatibility(
    const MatmulParams& attributes,
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const ttnn::Shape& a_shape,
    const ttnn::Shape& b_shape,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (!attributes.bcast_batch.has_value()) {
        return fmt::format("bcast_batch must be populated before matmul validation");
    }
    if (attributes.bcast_batch.value()) {
        if (get_batch_size(b_shape) != 1) {
            return fmt::format(
                "Batch-broadcast matmul requires input B batch size 1 (shapes BCMK*11KN=BCMN), got B batch size {}",
                get_batch_size(b_shape));
        }
    }

    // Validate batch dimensions for non-bcast matmul
    if (!attributes.bcast_batch.value()) {
        if (a_shape.rank() != b_shape.rank()) {
            return fmt::format(
                "Batched (non-bcast) matmul requires inputs of the same rank, got a_shape rank: {} vs b_shape rank: {}",
                a_shape.rank(),
                b_shape.rank());
        }

        // Check if in0 reuse optimization can be applied
        // This optimization keeps input A (batch=1) in L1 and reuses it across all input B batches
        // 1. Program config requirements: must use 1D mcast with specific settings
        // 2. Shape requirements: must be rank >= 3 (to have at least one batch dimension)
        // 3. Batch dimension requirement: all batch dimensions of input A must be size 1
        // 4. Memory layout requirement: inputs must not be sharded
        auto in0_reuse = [&]() {
            if (!std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                    chosen_program_config)) {
                return false;
            }
            const auto& config =
                std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
            if (config.fuse_batch || config.fused_activation.has_value() || config.mcast_in0) {
                return false;
            }
            if (a_shape.rank() < 3 || b_shape.rank() < 3) {
                return false;
            }
            for (auto j = 0; j < a_shape.rank() - 2; j++) {
                if (a_shape[j] != 1) {
                    return false;
                }
            }
            return !input_tensor_a.memory_config().is_sharded() && !input_tensor_b.memory_config().is_sharded();
        };

        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            if (!(a_shape[i] == b_shape[i] || (a_shape[i] == 1 && in0_reuse()))) {
                return fmt::format(
                    "bmm (non-bcast matmul) expects input tensors of shapes "
                    "BCMK*BCKN=BCMN or batch dimension {} mismatch: a={} vs b={} (dimension mismatch only allowed "
                    "when all batch dimensions of a are size 1 and using MatmulMultiCoreReuseMultiCast1DProgramConfig "
                    "with fuse_batch=false, fused_activation=none, mcast_in0=false, and non-sharded inputs for in0 "
                    "reuse optimization)",
                    i,
                    a_shape[i],
                    b_shape[i]);
            }
        }
    }
    return {};
}

// Input Count: checks there are normally exactly 2 inputs (activation + weight).
// Exception: the Mcast1D multi-tensor path (global_cb + DRAM-sharded weight) allows
// 1 activation + N weights, which must share shape/spec/layout/dtype.
std::string validate_matmul_input_count(
    const MatmulParams& attributes,
    const std::vector<TensorSpec>& input_tensors,
    const TensorSpec& input_tensor_b,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    if (std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            chosen_program_config) &&
        attributes.global_cb.has_value() && input_tensor_b.memory_config().is_sharded() &&
        (input_tensor_b.memory_config().buffer_type() == BufferType::DRAM)) {
        for (uint32_t i = 1; i < input_tensors.size(); ++i) {
            if (input_tensor_b.logical_shape() != input_tensors[i].logical_shape()) {
                return fmt::format(
                    "{}: for multi-tensor matmul, all weight tensors must have the same logical_shape, {} is not equal "
                    "to "
                    "{}",
                    config_name,
                    input_tensor_b.logical_shape(),
                    input_tensors[i].logical_shape());
            }
            if (input_tensor_b.padded_shape() != input_tensors[i].padded_shape()) {
                return fmt::format(
                    "{}: for multi-tensor matmul, all weight tensors must have the same padded_shape {} is not equal "
                    "to {}",
                    config_name,
                    input_tensor_b.padded_shape(),
                    input_tensors[i].padded_shape());
            }
            if (input_tensor_b != input_tensors[i]) {
                return fmt::format(
                    "{}: for multi-tensor matmul, all weight tensors must have the same tensor_spec {} is not equal to "
                    "{}",
                    config_name,
                    input_tensor_b,
                    input_tensors[i]);
            }
            if (input_tensor_b.layout() != input_tensors[i].layout()) {
                return fmt::format(
                    "{}: for multi-tensor matmul, all weight tensors must have the same layout {} is not equal to {}",
                    config_name,
                    input_tensor_b.layout(),
                    input_tensors[i].layout());
            }
            if (input_tensor_b.data_type() != input_tensors[i].data_type()) {
                return fmt::format(
                    "{}: for multi-tensor matmul, all weight tensors must have the same dtype {} is not equal to {}",
                    config_name,
                    input_tensor_b.data_type(),
                    input_tensors[i].data_type());
            }
        }
    } else {
        if (input_tensors.size() != 2) {
            return fmt::format("{}: Must have exactly 2 input tensors, got: {}", config_name, input_tensors.size());
        }
    }
    return {};
}

std::string validate_matmul_bias_shape(
    const std::optional<TensorSpec>& optional_bias,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape,
    const ttnn::Shape& b_shape_padded,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (!optional_bias.has_value()) {
        return {};
    }
    const auto& bias = optional_bias.value();
    auto bias_tile_shape = bias.tile().get_tile_shape();
    if (!(bias_tile_shape[0] == in0_tile.get_height() && bias_tile_shape[1] == in1_tile.get_width())) {
        return fmt::format(
            "Unsupported bias tile shape: bias tile ({}, {}) has to be (in0 tile height {}, in1 tile width {})",
            bias_tile_shape[0],
            bias_tile_shape[1],
            in0_tile.get_height(),
            in1_tile.get_width());
    }
    if (bias.layout() != Layout::TILE) {
        return fmt::format("Unsupported bias layout: {}, has to be TILE", bias.layout());
    }
    const auto& bias_shape = bias.logical_shape();
    const auto& bias_shape_padded = bias.padded_shape();
    uint32_t bias_batch_size = get_batch_size(bias_shape);
    if (bias_batch_size != 1) {
        return fmt::format("Unsupported bias shape: batch size must be 1, got {}", bias_batch_size);
    }
    // MatmulMultiCoreReuseProgramConfig fuses a full per-batch [M, N] bias block, so its height must
    // cover exactly M; every other config indexes a single bias tile-row.
    const bool is_reuse_config =
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseProgramConfig>(chosen_program_config);
    const uint32_t Mt = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const uint32_t expected_bias_height = (is_reuse_config ? Mt : 1) * in0_tile.get_height();
    if (bias_shape_padded[-2] != expected_bias_height) {
        return fmt::format(
            "Unsupported bias shape: padded second last dimension of bias, {}, not equal to expected bias height, "
            "{} (tile height {} x {} bias tile-row(s))",
            bias_shape_padded[-2],
            expected_bias_height,
            in0_tile.get_height(),
            is_reuse_config ? Mt : 1);
    }
    if (bias_shape_padded[-1] != b_shape_padded[-1]) {
        return fmt::format(
            "Unsupported bias shape: padded last dimension of bias, {}, not "
            "equal to second input's padded last "
            "dimension, {}.",
            bias_shape_padded[-1],
            b_shape_padded[-1]);
    }
    if (bias_shape[-1] < b_shape[-1]) {
        return fmt::format(
            "Unsupported bias shape: last dimension of bias, {}, not equal to "
            "or greater than second input's last "
            "dimension, {}.",
            bias_shape[-1],
            b_shape[-1]);
    }

    // Fused bias with a narrow in1 tile (width 16) and full-height in0 (32) is not supported
    // by the broadcast-row bias kernel path. Without this check the kernel hangs. See #42927.
    if (in0_tile.get_height() == TILE_HEIGHT && in1_tile.get_width() == 16) {
        const bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
        if (!in1_transpose_tile) {
            return fmt::format(
                "matmul fused bias does not support 32x16 narrow in1 tile (in0 tile height={}, in1 tile width={}). "
                "Use tile width 32 when bias is fused, or apply bias as a post-process add.",
                in0_tile.get_height(),
                in1_tile.get_width());
        }
    }
    return {};
}

// Untilize Output: untilize_out means "give me row-major output." Allowed only with an
// explicit BF16/FP32 output dtype on the Mcast1D config; this check runs for every
// config so it can reject untilize_out on every other config.
std::string validate_matmul_untilize_out(
    const MatmulParams& attributes, const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (!attributes.untilize_out) {
        return {};
    }
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    if (!attributes.output_dtype.has_value()) {
        return fmt::format("{}: Output dtype must be specified when untilize_out is true", config_name);
    }
    if (!((attributes.output_dtype.value() == DataType::BFLOAT16) ||
          (attributes.output_dtype.value() == DataType::FLOAT32))) {
        return fmt::format(
            "{}: Unsupported data type: {}, only BFLOAT16 and FLOAT32 are supported",
            config_name,
            attributes.output_dtype.value());
    }
    if (!(std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            chosen_program_config))) {
        return fmt::format(
            "{}: untilize_out is not supported for this program config, only supported for "
            "MatmulMultiCoreReuseMultiCast1DProgramConfig",
            config_name);
    }
    return {};
}

// ===========================================================================
// VALIDATIONS FOR MULTIPLE PROGRAM CONFIGS: each function here runs for several (but not
// all) program configs that need the same checks, branching internally per config.
// Messages are config-named.
// ===========================================================================

// Block/Subblock Configuration: runs for Reuse, Mcast2D, Mcast1D (MultiCore, DRAMSharded,
// BatchedDRAMSharded skip it).
std::string validate_matmul_block_and_subblock_configuration(
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    if (auto error = std::visit(
            [&](const auto& program_config) -> std::string {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (
                    std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreProgramConfig> ||
                    std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> ||
                    std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                    return {};
                }
                if constexpr (
                    std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig> ||
                    std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                    std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    const uint32_t Kt = a_shape_padded[-1] / in0_tile.get_width();
                    if (program_config.in0_block_w == 0) {
                        return fmt::format("{}: in0_block_w is 0, which is not valid", config_name);
                    }
                    if (Kt % program_config.in0_block_w != 0) {
                        return fmt::format(
                            "{}: Kt ({}) must be divisible by in0_block_w ({})",
                            config_name,
                            Kt,
                            program_config.in0_block_w);
                    }
                    if (program_config.out_subblock_h == 0) {
                        return fmt::format("{}: out_subblock_h is 0, which is not valid", config_name);
                    }
                    if (program_config.out_subblock_w == 0) {
                        return fmt::format("{}: out_subblock_w is 0, which is not valid", config_name);
                    }
                    if constexpr (
                        std::is_same_v<
                            ProgramConfigType,
                            operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                        std::is_same_v<
                            ProgramConfigType,
                            operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                        if (program_config.out_block_h == 0) {
                            return fmt::format("{}: out_block_h is 0, which is not valid", config_name);
                        }
                        if (program_config.out_block_w == 0) {
                            return fmt::format("{}: out_block_w is 0, which is not valid", config_name);
                        }
                        if (program_config.out_block_h % program_config.out_subblock_h != 0) {
                            return fmt::format(
                                "{}: out_block_h ({}) must be divisible by out_subblock_h ({})",
                                config_name,
                                program_config.out_block_h,
                                program_config.out_subblock_h);
                        }
                        if (program_config.out_block_w % program_config.out_subblock_w != 0) {
                            return fmt::format(
                                "{}: out_block_w ({}) must be divisible by out_subblock_w ({})",
                                config_name,
                                program_config.out_block_w,
                                program_config.out_subblock_w);
                        }
                        if (program_config.per_core_M % program_config.out_block_h != 0) {
                            return fmt::format(
                                "{}: per_core_M ({}) must be divisible by out_block_h ({})",
                                config_name,
                                program_config.per_core_M,
                                program_config.out_block_h);
                        }
                        if (program_config.per_core_N % program_config.out_block_w != 0) {
                            return fmt::format(
                                "{}: per_core_N ({}) must be divisible by out_block_w ({})",
                                config_name,
                                program_config.per_core_N,
                                program_config.out_block_w);
                        }
                    }
                    if constexpr (std::is_same_v<
                                      ProgramConfigType,
                                      operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                        if (program_config.per_core_M % program_config.out_subblock_h != 0) {
                            return fmt::format(
                                "{}: per_core_M ({}) must be divisible by out_subblock_h ({})",
                                config_name,
                                program_config.per_core_M,
                                program_config.out_subblock_h);
                        }
                        if (program_config.per_core_N % program_config.out_subblock_w != 0) {
                            return fmt::format(
                                "{}: per_core_N ({}) must be divisible by out_subblock_w ({})",
                                config_name,
                                program_config.per_core_N,
                                program_config.out_subblock_w);
                        }
                    }
                    if (!attributes.compute_kernel_config.has_value()) {
                        return fmt::format(
                            "{}: compute_kernel_config must be set for matmul subblock validation", config_name);
                    }
                    if (!attributes.output_tile.has_value()) {
                        return fmt::format("{}: output_tile must be set for matmul subblock validation", config_name);
                    }
                    const uint32_t available_reg_count = ttnn::get_dest_reg_count(
                        attributes.compute_kernel_config.value(), attributes.output_tile.value().get_tile_shape());
                    if (program_config.out_subblock_h * program_config.out_subblock_w > available_reg_count) {
                        return fmt::format(
                            "{}: out_subblock_w {} times out_subblock_h {} needs to be at most {} to fit in hardware",
                            config_name,
                            program_config.out_subblock_w,
                            program_config.out_subblock_h,
                            available_reg_count);
                    }
                }
                return {};
            },
            chosen_program_config);
        !error.empty()) {
        return error;
    }
    return {};
}

// Helper: checks in0_block_w / per_core_M / per_core_N are non-zero.
std::string validate_matmul_nonzero_block_dims(
    std::string_view config_name, std::size_t in0_block_w, std::size_t per_core_M, std::size_t per_core_N) {
    if (in0_block_w == 0) {
        return fmt::format("{}: in0_block_w is 0, which is not valid", config_name);
    }
    if (per_core_M == 0) {
        return fmt::format("{}: per_core_M is 0, which is not valid", config_name);
    }
    if (per_core_N == 0) {
        return fmt::format("{}: per_core_N is 0, which is not valid", config_name);
    }
    return {};
}

// Compute Grid & Per-Core Dims: skips MultiCore. For every other config it checks
// in0_block_w / per_core_M / per_core_N are non-zero (via the helper above); Reuse/
// Mcast2D/Mcast1D also check the program grid is non-zero and fits the device (Mcast1D
// gather_in0 skips that grid check — its grid comes from the input A shard grid).
std::string validate_matmul_compute_grid_and_per_core_dims(
    const DeviceDesc& device, const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const CoreCoord device_grid = device.grid;
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    if (auto error = std::visit(
            [&](const auto& program_config) -> std::string {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreProgramConfig>) {
                    return {};  // MultiCore: no program grid / block dims to check
                } else {
                    // Non-MultiCore. Grid-bounds check applies to Reuse/Mcast2D/Mcast1D only
                    // (DRAMSharded/BatchedDRAMSharded map to DRAM banks).
                    if constexpr (
                        std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig> ||
                        std::is_same_v<
                            ProgramConfigType,
                            operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                        std::is_same_v<
                            ProgramConfigType,
                            operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                        bool skip_grid_check = false;
                        if constexpr (std::is_same_v<
                                          ProgramConfigType,
                                          operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                            skip_grid_check = program_config.gather_in0;
                        }
                        if (!skip_grid_check) {
                            const auto& grid = program_config.compute_with_storage_grid_size;
                            if (!(grid.x > 0 && grid.y > 0)) {
                                return fmt::format(
                                    "{}: compute_with_storage_grid_size must be non-zero, got ({}, {})",
                                    config_name,
                                    grid.x,
                                    grid.y);
                            }
                            if (!(grid.x <= device_grid.x && grid.y <= device_grid.y)) {
                                return fmt::format(
                                    "{}: compute_with_storage_grid_size ({}, {}) must fit within device grid ({}, {})",
                                    config_name,
                                    grid.x,
                                    grid.y,
                                    device_grid.x,
                                    device_grid.y);
                            }
                        }
                    }
                    if (auto error = validate_matmul_nonzero_block_dims(
                            config_name,
                            program_config.in0_block_w,
                            program_config.per_core_M,
                            program_config.per_core_N);
                        !error.empty()) {
                        return error;
                    }
                    if constexpr (std::is_same_v<
                                      ProgramConfigType,
                                      operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                        if (auto error = dram_sharded_helpers::num_workers_per_dram_bank_error(
                                program_config.num_workers_per_dram_bank);
                            !error.empty()) {
                            return error;
                        }
                        if (!(program_config.num_workers_per_dram_bank == 1 || device.arch == tt::ARCH::BLACKHOLE)) {
                            return fmt::format(
                                "{}: num_workers_per_dram_bank > 1 is currently supported only on Blackhole",
                                config_name);
                        }
                    }
                }
                return {};
            },
            chosen_program_config);
        !error.empty()) {
        return error;
    }
    return {};
}

// Helper: an L1-sharded tensor's shard grid must fit within the given grid (DRAM skipped).
std::string check_tensor_in_grid(const TensorSpec& tensor, const CoreCoord& grid_size) {
    if (tensor.memory_config().is_sharded() && tensor.memory_config().buffer_type() != BufferType::DRAM) {
        const auto& shard_spec = tensor.memory_config().shard_spec().value();
        const auto& shard_grid = shard_spec.grid;
        if (!(grid_size.x > 0 && grid_size.y > 0)) {
            return fmt::format("compute grid size must be non-zero, got ({}, {})", grid_size.x, grid_size.y);
        }
        const CoreRange range(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1));
        if (!range.contains(shard_grid)) {
            return fmt::format(
                "Tensor shard spec grid {} must lie within compute grid ({}, {})",
                shard_grid,
                grid_size.x,
                grid_size.y);
        }
    }
    return {};
}

// Helper: an L1-sharded output's shard grid must fit within the given extent (DRAM skipped).
std::string check_output_shard_grid_within_extent(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::CoreCoord& extent,
    std::string_view config_name) {
    if (!output_mem_config.is_sharded() || output_mem_config.buffer_type() == tt::tt_metal::BufferType::DRAM) {
        return {};
    }
    if (!output_mem_config.shard_spec().has_value()) {
        return {};
    }
    const auto& shard_grid = output_mem_config.shard_spec().value().grid;
    if (!(extent.x > 0 && extent.y > 0)) {
        return fmt::format("{}: device grid extent must be non-zero, got ({}, {})", config_name, extent.x, extent.y);
    }
    const tt::tt_metal::CoreRange bbox(
        tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(extent.x - 1, extent.y - 1));
    if (!bbox.contains(shard_grid)) {
        return fmt::format("{}: output shard grid {} must lie within extent {}", config_name, shard_grid, extent);
    }
    return {};
}

// Work Distribution & Gather Ring: runs for Reuse/Mcast2D/Mcast1D. Checks output blocks
// fit the cores; for Mcast1D gather_in0 also checks the ring setup (A sharded, sub-device
// present, hop cores not overlapping).
std::string validate_matmul_work_distribution_and_gather_ring_topology(
    const DeviceDesc& device,
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    bool transpose_a,
    bool transpose_b,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    if (auto error = std::visit(
            [&](const auto& program_config) -> std::string {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<
                                  ProgramConfigType,
                                  operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    const tt::tt_metal::CoreCoord device_extent = device.grid;
                    const auto& grid = program_config.compute_with_storage_grid_size;
                    const auto Mt =
                        operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
                    const auto Nt = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                    const uint32_t per_core_M = program_config.per_core_M;
                    const uint32_t per_core_N = program_config.per_core_N;
                    const uint32_t num_cores = grid.x * grid.y;

                    if (program_config.gather_in0) {
                        if (transpose_a) {
                            return fmt::format("{}: transpose_a is not supported with gather_in0", config_name);
                        }
                        if (transpose_b) {
                            return fmt::format("{}: transpose_b is not supported with gather_in0", config_name);
                        }
                        if (!input_tensor_a.memory_config().is_sharded()) {
                            return fmt::format("{}: gather_in0 requires input A to be sharded", config_name);
                        }
                        if (!device.has_sub_devices) {
                            return fmt::format(
                                "{}: gather_in0 matmul requires at least one sub-device id on the device", config_name);
                        }
                        if (!program_config.hop_cores.empty()) {
                            const tt::tt_metal::CoreRangeSet& worker_cores =
                                input_tensor_a.memory_config().shard_spec().value().grid;
                            if (program_config.hop_cores.intersects(worker_cores)) {
                                return fmt::format(
                                    "{}: hop_cores must not overlap with input A shard grid. hop_cores={}, workers={}",
                                    config_name,
                                    program_config.hop_cores,
                                    worker_cores);
                            }
                        }
                        if (auto error =
                                check_output_shard_grid_within_extent(output_mem_config, device_extent, config_name);
                            !error.empty()) {
                            return error;
                        }
                    } else {
                        if (!program_config.hop_cores.empty()) {
                            return fmt::format(
                                "{}: Hop cores are not supported for any mode besides gather_in0.", config_name);
                        }
                        if (!(Mt > 0 && Nt > 0)) {
                            return fmt::format(
                                "{}: Mt and Nt must be greater than zero in tiles (got Mt={}, Nt={})",
                                config_name,
                                Mt,
                                Nt);
                        }
                        const uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
                        const uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
                        const uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                        if (num_blocks_total > num_cores) {
                            return fmt::format(
                                "{}: Number of blocks exceeds number of cores: {} blocks > {} cores",
                                config_name,
                                num_blocks_total,
                                num_cores);
                        }
                        if (program_config.mcast_in0) {
                            if (num_blocks_y != 1) {
                                return fmt::format(
                                    "{}: mcast_in0 requires M ({}) to fit within a single per_core_M block ({}), got "
                                    "num_blocks_y={}",
                                    config_name,
                                    Mt,
                                    per_core_M,
                                    num_blocks_y);
                            }
                        } else {
                            if (num_blocks_x != 1) {
                                return fmt::format(
                                    "{}: mcast_in1 requires N ({}) to fit within a single per_core_N block ({}), got "
                                    "num_blocks_x={}. A single in1 sender multicasts one per_core_N-wide weight slice "
                                    "to "
                                    "the whole grid, so multi-column N is not supported here; use "
                                    "MatmulMultiCoreReuseMultiCastProgramConfig (the 2D factory) for multi-column N.",
                                    config_name,
                                    Nt,
                                    per_core_N,
                                    num_blocks_x);
                            }
                            if (per_core_M > Mt) {
                                return fmt::format(
                                    "{}: per_core_M ({}) exceeds Mt ({}). Each core would compute more output row "
                                    "tiles "
                                    "than the tensor has. Reduce per_core_M to at most Mt.",
                                    config_name,
                                    per_core_M,
                                    Mt);
                            }
                            const uint32_t logical_blocks_w = ((Nt - 1) / program_config.out_block_w) + 1;
                            const uint32_t physical_blocks_w = per_core_N / program_config.out_block_w;
                            if (logical_blocks_w != physical_blocks_w) {
                                return fmt::format(
                                    "{}: mcast_in1 requires the logical N tail to be in the final internal W block; "
                                    "got N={}, per_core_N={}, out_block_w={} (logical blocks={}, physical blocks={}). "
                                    "Reduce per_core_N or increase out_block_w.",
                                    config_name,
                                    Nt,
                                    per_core_N,
                                    program_config.out_block_w,
                                    logical_blocks_w,
                                    physical_blocks_w);
                            }
                            if (num_blocks_y == 1) {
                                const uint32_t logical_blocks_h = ((Mt - 1) / program_config.out_block_h) + 1;
                                const uint32_t physical_blocks_h = per_core_M / program_config.out_block_h;
                                if (logical_blocks_h != physical_blocks_h) {
                                    return fmt::format(
                                        "{}: a single-Y mcast_in1 sender requires the logical M tail to be in the "
                                        "final "
                                        "internal H block; got M={}, per_core_M={}, out_block_h={} (logical blocks={}, "
                                        "physical blocks={}). Reduce per_core_M or increase out_block_h.",
                                        config_name,
                                        Mt,
                                        per_core_M,
                                        program_config.out_block_h,
                                        logical_blocks_h,
                                        physical_blocks_h);
                                }
                                if (!(Mt % program_config.out_block_h == 0 || physical_blocks_h == 1)) {
                                    return fmt::format(
                                        "{}: a single-Y mcast_in1 sender supports a partial final H block only when "
                                        "per_core_M contains one internal H block; got M={}, per_core_M={}, "
                                        "out_block_h={} "
                                        "(physical blocks={}).",
                                        config_name,
                                        Mt,
                                        per_core_M,
                                        program_config.out_block_h,
                                        physical_blocks_h);
                                }
                            }
                        }
                        if (auto error = check_output_shard_grid_within_extent(output_mem_config, grid, config_name);
                            !error.empty()) {
                            return error;
                        }
                    }
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    const auto& grid = program_config.compute_with_storage_grid_size;
                    const auto Mt =
                        operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
                    const auto Nt = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                    if (!(Mt > 0 && Nt > 0)) {
                        return fmt::format(
                            "{}: Mt and Nt must be greater than zero in tiles (got Mt={}, Nt={})", config_name, Mt, Nt);
                    }
                    if (program_config.per_core_M > Mt) {
                        return fmt::format(
                            "{}: per_core_M ({}) exceeds Mt ({}). Each core would compute more output row tiles "
                            "than the tensor has. Reduce per_core_M to at most Mt.",
                            config_name,
                            program_config.per_core_M,
                            Mt);
                    }
                    uint32_t num_blocks_y = ((Mt - 1) / program_config.per_core_M) + 1;
                    uint32_t num_blocks_x = ((Nt - 1) / program_config.per_core_N) + 1;
                    if (program_config.transpose_mcast) {
                        std::swap(num_blocks_x, num_blocks_y);
                    }
                    if (num_blocks_x > grid.x) {
                        return fmt::format(
                            "{}: Num output blocks along x ({}) must be smaller than or equal to the number of columns "
                            "in "
                            "compute grid ({})!",
                            config_name,
                            num_blocks_x,
                            grid.x);
                    }
                    if (num_blocks_y > grid.y) {
                        return fmt::format(
                            "{}: Num output blocks along y ({}) must be smaller than or equal to the number of rows in "
                            "compute "
                            "grid ({})!",
                            config_name,
                            num_blocks_y,
                            grid.y);
                    }
                    if (auto error = check_output_shard_grid_within_extent(output_mem_config, grid, config_name);
                        !error.empty()) {
                        return error;
                    }
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                    // The factory selects all_cores from the first available shard spec: in0, then in1,
                    // then output. Any of those can produce an offset grid (e.g. column 1 in a fused
                    // chain). Use the device grid as the extent whenever any operand is sharded so we
                    // don't incorrectly reject those grids against the origin-anchored config rect.
                    const auto device_extent = device.grid;
                    const bool any_sharded = input_tensor_a.memory_config().is_sharded() ||
                                             input_tensor_b.memory_config().is_sharded() ||
                                             output_mem_config.is_sharded();
                    const auto effective_extent =
                        any_sharded ? device_extent : program_config.compute_with_storage_grid_size;
                    if (auto error =
                            check_output_shard_grid_within_extent(output_mem_config, effective_extent, config_name);
                        !error.empty()) {
                        return error;
                    }
                } else {
                    (void)transpose_a;
                    (void)transpose_b;
                }
                return {};
            },
            chosen_program_config);
        !error.empty()) {
        return error;
    }
    return {};
}

// Sharded Operand Grids: Reuse config only. Each L1-sharded input's shard grid must fit
// within the device grid. Non-sharded inputs and DRAM inputs are not checked.
std::string validate_matmul_sharded_operand_grids_within_program_compute_grid(
    const DeviceDesc& device,
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (auto error = std::visit(
            [&](const auto& program_config) -> std::string {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::
                                  is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                    // When an input is sharded, the factory uses shard_spec.grid directly as all_cores
                    // and ignores compute_with_storage_grid_size entirely. Validating the shard grid
                    // against the origin-anchored compute_with_storage_grid_size rectangle incorrectly
                    // rejects grids that don't start at (0,0) (e.g. column 1 in a multi-chain fused op).
                    // The only physical constraint is that the shard grid fits within the device grid.
                    // Non-sharded inputs are not checked here (the check only applies to L1-sharded tensors).
                    const auto& config_grid = program_config.compute_with_storage_grid_size;
                    const auto device_grid = device.grid;
                    auto effective_grid_a = input_tensor_a.memory_config().is_sharded() ? device_grid : config_grid;
                    auto effective_grid_b = input_tensor_b.memory_config().is_sharded() ? device_grid : config_grid;
                    if (auto error = check_tensor_in_grid(input_tensor_a, effective_grid_a); !error.empty()) {
                        return error;
                    }
                    if (auto error = check_tensor_in_grid(input_tensor_b, effective_grid_b); !error.empty()) {
                        return error;
                    }
                }
                return {};
            },
            chosen_program_config);
        !error.empty()) {
        return error;
    }
    return {};
}

// Output Block Divisibility: Reuse config only. If an input is sharded across N cores,
// the total number of output blocks must be a multiple of N so the work splits evenly
// across those cores; otherwise the program factory can't distribute it and fails.
std::string validate_matmul_reuse_sharded_output_block_divisibility(
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (auto error = std::visit(
            [&](const auto& program_config) -> std::string {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::
                                  is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                    // Mirror the shard_spec priority in MatmulMultiCoreReuseOptimizedProgramFactory::create_descriptor:
                    // when in0 is L1-sharded its shard grid becomes the kernel grid; in1's grid is only consulted when
                    // in0 is not sharded. num_output_blocks must divide evenly across that grid or the factory fatals.
                    const TensorSpec* sharded = nullptr;
                    if (input_tensor_a.memory_config().is_sharded() &&
                        input_tensor_a.memory_config().buffer_type() != BufferType::DRAM) {
                        sharded = &input_tensor_a;
                    } else if (
                        input_tensor_b.memory_config().is_sharded() &&
                        input_tensor_b.memory_config().buffer_type() != BufferType::DRAM) {
                        sharded = &input_tensor_b;
                    }
                    if (sharded == nullptr) {
                        return {};
                    }
                    const uint32_t B = get_batch_size(a_shape_padded);
                    const uint32_t Mt = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, false);
                    const uint32_t Nt = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                    const uint32_t num_output_blocks =
                        (B * Mt / program_config.per_core_M) * (Nt / program_config.per_core_N);
                    const uint32_t num_cores = sharded->memory_config().shard_spec().value().grid.num_cores();
                    if (num_output_blocks % num_cores != 0) {
                        return fmt::format(
                            "MatmulMultiCoreReuseProgramConfig: num_output_blocks ({}) must be evenly divisible by the "
                            "number of cores in the input shard grid ({})",
                            num_output_blocks,
                            num_cores);
                    }
                }
                return {};
            },
            chosen_program_config);
        !error.empty()) {
        return error;
    }
    return {};
}

// Helper: cross-validate a DRAM-sender global_cb's geometry against the matmul + weight shape.
// These catch silent-hang configs where the matmul reads more in1 pages than the prefetcher
// pushes (e.g. activation K padded past weight K). Gated by the caller on the DRAM-sender path
// because the worker-sender variant predates this work and uses different sizing/ordering
// conventions (no bank IDs; gcb_size = N * max_tile_size).
std::string validate_dram_sender_global_cb_gather_in0_geometry(
    const tt::tt_metal::experimental::GlobalCircularBuffer& gcb,
    const TensorSpec& input_tensor_a,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    const uint32_t ring_size = input_tensor_a.memory_config().shard_spec().value().grid.num_cores();
    const uint32_t weight_K_tiles = b_shape_padded[-2] / in1_tile.get_height();
    const uint32_t weight_N_tiles = b_shape_padded[-1] / in1_tile.get_width();
    const uint32_t num_senders = gcb.sender_cores().num_cores();
    const uint32_t num_recv = gcb.receiver_cores().num_cores();
    const uint32_t recv_per_bank = static_cast<uint32_t>(program_config.num_global_cb_receivers);

    if (!(num_senders > 0 && num_recv == num_senders * recv_per_bank)) {
        return fmt::format(
            "global_cb receiver count ({}) must equal num_senders ({}) * "
            "num_global_cb_receivers ({})",
            num_recv,
            num_senders,
            recv_per_bank);
    }
    if (num_recv != ring_size) {
        return fmt::format(
            "global_cb receiver count ({}) must equal in0 (activation) ring_size "
            "({} = num_cores of in0.shard_spec.grid). Receivers and matmul workers "
            "must be the same set of cores.",
            num_recv,
            ring_size);
    }

    // Semantic check: bank b must push to exactly the receivers at ring
    // positions [b*recv_per_bank, (b+1)*recv_per_bank). If satisfied, the
    // bank-to-receivers union also equals the activation grid as a set, so
    // we don't need a separate set-equality assertion (CoreRangeSet::operator==
    // compares ranges literally, which is brittle when one side is merged
    // into rectangles and the other is a flat list of single-core ranges).
    const auto& act_grid = input_tensor_a.memory_config().shard_spec().value().grid;
    const auto ring_walk = tt::tt_metal::corerange_to_cores(act_grid, std::nullopt, /*row_wise=*/true);
    const auto& mapping = gcb.sender_receiver_core_mapping();
    if (mapping.size() * recv_per_bank != ring_walk.size()) {
        return fmt::format(
            "global_cb sender_receiver mapping ({} senders * {} receivers each) "
            "doesn't cover the matmul ring ({} cores)",
            mapping.size(),
            recv_per_bank,
            ring_walk.size());
    }
    for (size_t bank_idx = 0; bank_idx < mapping.size(); ++bank_idx) {
        const auto bank_recvs =
            tt::tt_metal::corerange_to_cores(mapping[bank_idx].second, std::nullopt, /*row_wise=*/true);
        if (bank_recvs.size() != recv_per_bank) {
            return fmt::format(
                "Sender at bank index {} owns {} receivers; expected "
                "num_global_cb_receivers={}",
                bank_idx,
                bank_recvs.size(),
                recv_per_bank);
        }
        for (size_t k = 0; k < recv_per_bank; ++k) {
            const size_t ring_pos = bank_idx * recv_per_bank + k;
            if (bank_recvs[k] != ring_walk[ring_pos]) {
                return fmt::format(
                    "global_cb bank {}'s receiver at index {} is core {} but the "
                    "matmul ring walk expects core {} at ring position {}. The "
                    "bank-to-receivers mapping must place bank b's receivers at "
                    "ring positions [b*num_global_cb_receivers, (b+1)*num_global_cb_receivers).",
                    bank_idx,
                    k,
                    bank_recvs[k],
                    ring_walk[ring_pos],
                    ring_pos);
            }
        }
    }
    if (weight_K_tiles % ring_size != 0) {
        return fmt::format(
            "Weight K must be divisible by ring_size in tiles for gather_in0 + global_cb. "
            "Got weight_K_tiles={}, ring_size={} (remainder={}). The activation grid would "
            "pad K past the weight K, and the matmul would wait forever for in1 pages the "
            "prefetcher never pushes.",
            weight_K_tiles,
            ring_size,
            weight_K_tiles % ring_size);
    }
    if (weight_N_tiles % num_senders != 0) {
        return fmt::format(
            "Weight N ({} tiles) must be divisible by num_senders ({}) so it shards "
            "evenly across the DRAM banks the global_cb senders cover",
            weight_N_tiles,
            num_senders);
    }
    const uint32_t per_bank_N_tiles = weight_N_tiles / num_senders;
    if (per_bank_N_tiles % recv_per_bank != 0) {
        return fmt::format(
            "Weight per-bank N ({} tiles) must be divisible by num_global_cb_receivers ({})",
            per_bank_N_tiles,
            recv_per_bank);
    }
    const uint32_t per_recv_N_tiles = per_bank_N_tiles / recv_per_bank;
    if (per_recv_N_tiles != program_config.per_core_N) {
        return fmt::format(
            "Matmul per_core_N ({}) must equal weight per-receiver N ({} = per_bank_N_tiles {} "
            "/ num_global_cb_receivers {})",
            program_config.per_core_N,
            per_recv_N_tiles,
            per_bank_N_tiles,
            recv_per_bank);
    }
    return {};
}

std::string validate_dram_sender_global_cb_gather_in0_geometry_recv_contig(
    const tt::tt_metal::experimental::GlobalCircularBuffer& gcb,
    const TensorSpec& input_tensor_a,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    const uint32_t ring_size = input_tensor_a.memory_config().shard_spec().value().grid.num_cores();
    const uint32_t num_recv = gcb.receiver_cores().num_cores();
    if (num_recv != ring_size) {
        return fmt::format(
            "global_cb receiver count ({}) must equal in0 (activation) ring_size ({}). Receivers and matmul "
            "workers must be the same set of cores.",
            num_recv,
            ring_size);
    }

    const uint32_t weight_K_tiles = b_shape_padded[-2] / in1_tile.get_height();
    const uint32_t weight_N_tiles = b_shape_padded[-1] / in1_tile.get_width();
    if (weight_K_tiles % ring_size != 0) {
        return fmt::format(
            "Weight K ({} tiles) must be divisible by ring_size ({}) for receiver-contiguous gather_in0 + "
            "global_cb (remainder {}). The activation grid pads K past the weight K and the matmul would "
            "wait forever for in1 pages the prefetcher never pushes.",
            weight_K_tiles,
            ring_size,
            weight_K_tiles % ring_size);
    }
    if (weight_N_tiles % ring_size != 0) {
        return fmt::format(
            "Weight N ({} tiles) must be divisible by ring_size ({}) for receiver-contiguous gather_in0 + global_cb",
            weight_N_tiles,
            ring_size);
    }
    const uint32_t per_recv_N_tiles = weight_N_tiles / ring_size;
    if (per_recv_N_tiles != program_config.per_core_N) {
        return fmt::format(
            "Matmul per_core_N ({}) must equal weight per-receiver N ({} = N_tiles {} / ring_size {}); otherwise "
            "the matmul's in1 page size disagrees with what the recv-contig prefetcher pushes.",
            program_config.per_core_N,
            per_recv_N_tiles,
            weight_N_tiles,
            ring_size);
    }
    return {};
}

std::string validate_dram_sender_global_cb_mcast_in0_geometry(
    const tt::tt_metal::experimental::GlobalCircularBuffer& gcb,
    const TensorSpec& input_tensor_b,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    if (tt::tt_metal::experimental::sender_core_type(gcb) != tt::tt_metal::experimental::SenderCoreType::Dram) {
        return fmt::format("mcast_in0 global_cb requires programmable DRAM senders");
    }
    if (!(program_config.out_block_h == program_config.per_core_M &&
          program_config.out_block_w == program_config.per_core_N)) {
        return fmt::format(
            "mcast_in0 global_cb requires one output block per worker: out_block_h ({}) must equal per_core_M ({}) "
            "and out_block_w ({}) must equal per_core_N ({})",
            program_config.out_block_h,
            program_config.per_core_M,
            program_config.out_block_w,
            program_config.per_core_N);
    }

    // The weight ↔ matmul cross-checks are the prefetcher's (they read the weight's buffer); the device op runs
    // them after these.

    // GCB-window guard specific to this op: the mcast reader streams K-blocks through a remote-CB
    // window, so the GCB has to hold at least a double buffer of them. The window itself is floored to
    // whole pages when the CB is created, so a size that is not an exact multiple is fine — the leftover
    // bytes are simply unused.
    const uint32_t in1_block_size_bytes =
        program_config.in0_block_w * program_config.per_core_N *
        in1_tile.get_tile_size(tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.data_type()));
    const uint32_t resident_blocks = gcb.size() / in1_block_size_bytes;
    if (resident_blocks < 2) {
        return fmt::format(
            "mcast_in0 global_cb requires a two-page streaming window: size {} holds {} whole in1 K-block pages of {} "
            "B, "
            "need at least 2",
            gcb.size(),
            resident_blocks,
            in1_block_size_bytes);
    }
    return {};
}

// Sub-Device Worker Grid: Mcast1D on a sub-device (non-gather). Checks the matmul grid
// fits on the sub-device's cores.
std::string validate_matmul_mcast1d_subdevice_worker_grid(
    const DeviceDesc& device,
    const MatmulParams& attributes,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    // matmul_multicore_reuse_mcast_1d (both program- and descriptor-based) targets a single
    // bounding-box rectangle for the in0/in1 multicast and expects the sub-device's worker
    // cores to form one contiguous row-major rectangle. Reject non-rectangular sub-device
    // grids early with a clear message.
    if (std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            chosen_program_config) &&
        attributes.sub_device_id.has_value()) {
        const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
        const auto& program_config_1d =
            std::get<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
        if (!program_config_1d.gather_in0) {
            const auto& sub_device_cores = device.sub_device_workers.value();
            auto bbox = sub_device_cores.bounding_box();
            if (sub_device_cores.num_cores() != bbox.size()) {
                return fmt::format(
                    "{}: matmul_multicore_reuse_mcast_1d only supports rectangular sub-device worker grids. "
                    "Got sub-device worker cores: {} (bounding box: {})",
                    config_name,
                    sub_device_cores,
                    bbox);
            }
            auto grid_size_1d = program_config_1d.allowed_worker_cores.value().bounding_box().grid_size();
            if (!(bbox.start_coord.x + grid_size_1d.x - 1 <= bbox.end_coord.x &&
                  bbox.start_coord.y + grid_size_1d.y - 1 <= bbox.end_coord.y)) {
                return fmt::format(
                    "{}: matmul grid_size {} anchored at sub-device start {} "
                    "extends past the sub-device's worker bounding box {}",
                    config_name,
                    grid_size_1d,
                    bbox.start_coord,
                    bbox);
            }
        }
    }
    return {};
}

// Helper: input A and output must agree on buffer type + memory layout. Used by
// Reuse/Mcast2D/Mcast1D/DRAMSharded/BatchedDRAMSharded.
std::string validate_input_a_output_mem_config_match(
    std::string_view config_name,
    const TensorSpec& input_tensor_a,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    if (input_tensor_a.memory_config().buffer_type() != output_mem_config.buffer_type()) {
        return fmt::format(
            "{}: input A and output buffer types must match, got input: {} vs output: {}",
            config_name,
            input_tensor_a.memory_config().buffer_type(),
            output_mem_config.buffer_type());
    }
    if (input_tensor_a.memory_config().memory_layout() != output_mem_config.memory_layout()) {
        return fmt::format(
            "{}: input A and output memory layouts must match, got input: {} vs output: {}",
            config_name,
            input_tensor_a.memory_config().memory_layout(),
            output_mem_config.memory_layout());
    }
    return {};
}

// Helper: output subblock/block width must divide per_core_N. Used by
// Mcast2D/Mcast1D (Reuse keeps its own inline variant).
std::string validate_output_subblock_block_divides_per_core_n(
    std::string_view config_name,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t out_block_w,
    uint32_t out_block_h,
    uint32_t per_core_N) {
    if (!(out_subblock_w == per_core_N || out_subblock_h == 1)) {
        return fmt::format(
            "{}: out_subblock_w ({}) must equal per_core_N ({}) or out_subblock_h ({}) must be 1",
            config_name,
            out_subblock_w,
            per_core_N,
            out_subblock_h);
    }
    if (!(out_block_w == per_core_N || out_block_h == 1)) {
        return fmt::format(
            "{}: out_block_w ({}) must equal per_core_N ({}) or out_block_h ({}) must be 1",
            config_name,
            out_block_w,
            per_core_N,
            out_block_h);
    }
    return {};
}

// ===========================================================================
// PROGRAM CONFIG SPECIFIC VALIDATIONS: one function per program config, holding the
// checks that fire for that config only. Dispatched from program_config_error via one
// std::visit.
// ===========================================================================

// MultiCore config: the un-optimized fallback. Rejects tiny outer tiles and requires
// all operands + output to be INTERLEAVED.
std::string validate_matmul_multicore_config(
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const MatmulParams& attributes,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile) {
    constexpr auto config_name = ttsl::get_type_name<operations::matmul::MatmulMultiCoreProgramConfig>();
    const bool uses_tiny_outer_tile = (in0_tile.get_height() != TILE_HEIGHT || in1_tile.get_width() != TILE_WIDTH);
    if (uses_tiny_outer_tile) {
        return fmt::format(
            "{}: matmul with non-optimized program config does not support tiny tile "
            "(in0 tile height={}, in1 tile width={}, expected TILE_HEIGHT={}, TILE_WIDTH={})",
            config_name,
            in0_tile.get_height(),
            in1_tile.get_width(),
            TILE_HEIGHT,
            TILE_WIDTH);
    }
    if (input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return fmt::format(
            "{}: Input A memory layout must be INTERLEAVED, got: {}",
            config_name,
            input_tensor_a.memory_config().memory_layout());
    }
    if (input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return fmt::format(
            "{}: Input B memory layout must be INTERLEAVED, got: {}",
            config_name,
            input_tensor_b.memory_config().memory_layout());
    }
    if (attributes.output_mem_config.memory_layout() != TensorMemoryLayout::INTERLEAVED) {
        return fmt::format(
            "{}: Output memory layout must be INTERLEAVED, got: {}",
            config_name,
            attributes.output_mem_config.memory_layout());
    }
    return {};
}

// DRAMSharded config: in0 width-sharded in L1, in1 width-sharded in DRAM; height must
// be a single tile (M == 1) and K/shard dims divide in0_block_w.
std::string validate_matmul_dram_sharded_config(
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig& program_config) {
    const auto config_name = ttsl::get_type_name(program_config);

    // The DRAM-sharded mcast path is not yet validated for tile_h < 16 — it hangs for
    // all tile_h in {1,2,4,8} regardless of dtype or tile_w. Only tile_h in {16, 32} is
    // currently supported on this path. See #42927.
    if (in0_tile.get_height() < 16) {
        return fmt::format(
            "{}: matmul tiny-tile with tile_h {} is not currently supported on the DRAM-sharded mcast "
            "program config path (requires tile_h >= 16); see issue #42927",
            config_name,
            in0_tile.get_height());
    }

    if (!input_tensor_a.memory_config().is_sharded()) {
        return fmt::format("{}: Input tensor A must be sharded for DRAM sharded program config", config_name);
    }
    if (!attributes.output_mem_config.is_sharded()) {
        return fmt::format("{}: Output memory config must be sharded for DRAM sharded program config", config_name);
    }
    if (input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
        return fmt::format(
            "{}: Input A memory layout must be WIDTH_SHARDED, got: {}",
            config_name,
            input_tensor_a.memory_config().memory_layout());
    }
    if (auto error =
            validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
        !error.empty()) {
        return error;
    }
    if (input_tensor_a.memory_config().shard_spec().value().orientation != ShardOrientation::ROW_MAJOR) {
        return fmt::format(
            "{}: Input A shard orientation must be ROW_MAJOR, got: {}",
            config_name,
            input_tensor_a.memory_config().shard_spec().value().orientation);
    }
    const auto M = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
    uint32_t per_core_M = program_config.per_core_M;
    auto shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;

    // No padding
    if (M != per_core_M) {
        return fmt::format("{}: M ({}) must equal per_core_M ({})", config_name, M, per_core_M);
    }
    if (M != 1) {
        return fmt::format("{}: currently only support in0 tensor height of tile height", config_name);
    }
    if (per_core_M != (shard_shape[0] / in0_tile.get_height())) {
        return fmt::format(
            "{}: per_core_M ({}) must equal shard_shape[0] / in0_tile.get_height() ({})",
            config_name,
            per_core_M,
            (shard_shape[0] / in0_tile.get_height()));
    }
    if (K % program_config.in0_block_w != 0) {
        return fmt::format(
            "{}: K ({}) must be divisible by in0_block_w ({})", config_name, K, program_config.in0_block_w);
    }
    // A block is either a fraction of one storage shard or a whole number of consecutive shards.
    const uint32_t in0_shard_width_tiles = shard_shape[1] / in0_tile.get_width();
    if (!(in0_shard_width_tiles % program_config.in0_block_w == 0 ||
          program_config.in0_block_w % in0_shard_width_tiles == 0)) {
        return fmt::format(
            "{}: shard_shape[1] / in0_tile.get_width() ({}) and in0_block_w ({}) must divide one another",
            config_name,
            in0_shard_width_tiles,
            program_config.in0_block_w);
    }

    // tensor in1
    if (input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
        return fmt::format(
            "{}: Input B memory layout must be WIDTH_SHARDED, got: {}",
            config_name,
            input_tensor_b.memory_config().memory_layout());
    }
    return {};
}

// BatchedDRAMSharded config: [1,B,M,K] x [1,B,K,N]: A height-sharded in L1, B height-
// sharded in DRAM, output height-sharded in L1; contracted dim divides in0_block_w.
std::string validate_matmul_batched_dram_sharded_config(
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig& program_config) {
    const auto config_name = ttsl::get_type_name(program_config);
    // Batch-sharded DRAM matmul validations
    // For batched matmul: [1, B, M, K] x [1, B, K, N] = [1, B, M, N]
    // Sharded by batch dimension - each worker handles B/num_workers complete matmuls
    // Input A: HEIGHT_SHARDED in L1 (batch-sharded, each core has B/num_workers complete [M, K] matrices)
    // Input B: HEIGHT_SHARDED in DRAM (batch-sharded, each bank has B/num_workers complete [K, N] matrices)
    // Output: HEIGHT_SHARDED in L1 (batch-sharded, each core outputs B/num_workers complete [M, N] matrices)
    if (!input_tensor_a.memory_config().is_sharded()) {
        return fmt::format("{}: Input tensor A must be sharded for batch-sharded DRAM matmul", config_name);
    }
    if (!attributes.output_mem_config.is_sharded()) {
        return fmt::format("{}: Output memory config must be sharded for batch-sharded DRAM matmul", config_name);
    }
    if (input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return fmt::format(
            "{}: Input A memory layout must be HEIGHT_SHARDED for batch-sharded DRAM matmul, got: {}",
            config_name,
            input_tensor_a.memory_config().memory_layout());
    }
    if (auto error =
            validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
        !error.empty()) {
        return error;
    }
    if (input_tensor_a.memory_config().shard_spec().value().orientation != ShardOrientation::ROW_MAJOR) {
        return fmt::format(
            "{}: Input A shard orientation must be ROW_MAJOR, got: {}",
            config_name,
            input_tensor_a.memory_config().shard_spec().value().orientation);
    }

    // For batch sharding, the contracted dimension (N in A, N in B) must be divisible by in0_block_w
    const auto N_dim = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);  // K dim of A = N
    if (N_dim % program_config.in0_block_w != 0) {
        return fmt::format(
            "{}: N dimension ({}) must be divisible by in0_block_w ({})",
            config_name,
            N_dim,
            program_config.in0_block_w);
    }

    // tensor in1: HEIGHT_SHARDED in DRAM (batch-sharded)
    if (input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return fmt::format(
            "{}: Input B memory layout must be HEIGHT_SHARDED for batch-sharded DRAM matmul, got: {}",
            config_name,
            input_tensor_b.memory_config().memory_layout());
    }
    return {};
}

// Mcast2D config: block-sharded 2D multicast. Validates that sharded input A, input B,
// and the output have layouts, grids, and orientations consistent with a 2D multicast.
std::string validate_matmul_mcast2d_config(
    const DeviceDesc& device,
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config) {
    using namespace tt;  // BufferType/div_up were unqualified in the original std::visit scope
    const auto config_name = ttsl::get_type_name(program_config);
    const tt::tt_metal::CoreCoord device_grid = device.grid;
    if (auto error = check_tensor_in_grid(input_tensor_a, device_grid); !error.empty()) {
        return error;
    }
    if (auto error = check_tensor_in_grid(input_tensor_b, device_grid); !error.empty()) {
        return error;
    }
    if (input_tensor_a.memory_config().is_sharded()) {
        if (!program_config.fuse_batch) {
            return fmt::format("{}: Batch fusion is required when input A is sharded", config_name);
        }
        auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
        const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
        uint32_t per_core_M = program_config.per_core_M;
        auto shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;

        if (!(tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
              tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED)) {
            return fmt::format("{}: Unsupported memory layout {}.", config_name, tensor_a_memory_layout);
        }

        if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            if (program_config.transpose_mcast) {
                if (input_tensor_a.memory_config().shard_spec().value().orientation != ShardOrientation::COL_MAJOR) {
                    return fmt::format(
                        "{}: Input tensor A must have COL_MAJOR shard orientation for transpose MCAST, got: {}",
                        config_name,
                        input_tensor_a.memory_config().shard_spec().value().orientation);
                }
            } else {
                if (input_tensor_a.memory_config().shard_spec().value().orientation != ShardOrientation::ROW_MAJOR) {
                    return fmt::format(
                        "{}: Input tensor A must have ROW_MAJOR shard orientation for non-transpose MCAST, got: {}",
                        config_name,
                        input_tensor_a.memory_config().shard_spec().value().orientation);
                }
            }
            if (attributes.output_mem_config.is_sharded()) {
                if (auto error = validate_input_a_output_mem_config_match(
                        config_name, input_tensor_a, attributes.output_mem_config);
                    !error.empty()) {
                    return error;
                }
            }

        } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            if (program_config.transpose_mcast) {
                return fmt::format("{}: Transpose MCAST not supported with HEIGHT_SHARDED layout", config_name);
            }
            if (K != program_config.in0_block_w) {
                return fmt::format(
                    "{}: K ({}) must equal in0_block_w ({})", config_name, K, program_config.in0_block_w);
            }
            if (program_config.in0_block_w != (shard_shape[1] / in0_tile.get_width())) {
                return fmt::format(
                    "{}: in0_block_w ({}) must equal shard_shape[1] / in0_tile.get_width() ({})",
                    config_name,
                    program_config.in0_block_w,
                    (shard_shape[1] / in0_tile.get_width()));
            }
            if (!(input_tensor_a.memory_config().shard_spec()->grid.bounding_box().start_coord.x ==
                  input_tensor_a.memory_config().shard_spec()->grid.bounding_box().end_coord.x)) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input A must have a single-column shard grid (got x={} to x={}); use "
                    "MatmulMultiCoreReuseProgramConfig for multi-column HEIGHT_SHARDED inputs",
                    config_name,
                    input_tensor_a.memory_config().shard_spec()->grid.bounding_box().start_coord.x,
                    input_tensor_a.memory_config().shard_spec()->grid.bounding_box().end_coord.x);
            }
        }

        if (per_core_M != (shard_shape[0] / in0_tile.get_height())) {
            return fmt::format(
                "{}: per_core_M ({}) must equal shard_shape[0] / in0_tile.get_height() ({})",
                config_name,
                per_core_M,
                (shard_shape[0] / in0_tile.get_height()));
        }
        if ((shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w != 0) {
            return fmt::format(
                "{}: shard_shape[1] / in0_tile.get_width() ({}) must be divisible by in0_block_w ({})",
                config_name,
                (shard_shape[1] / in0_tile.get_width()),
                program_config.in0_block_w);
        }
    }

    if (input_tensor_b.memory_config().is_sharded()) {
        if (program_config.transpose_mcast) {
            return fmt::format("{}: Transpose MCAST not supported when input B is sharded", config_name);
        }
        auto tensor_b_memory_layout = input_tensor_b.memory_config().memory_layout();
        // ND_SHARDED in1 in DRAM is read via the generic TensorAccessor path: the program
        // factory's in1_is_sharded only covers WIDTH/HEIGHT, so ND falls through to the
        // interleaved-style reader, which addresses the NdShardSpec layout from the accessor
        // args. The width/height-specific validation below is gated on those layouts, so ND
        // DRAM in1 skips it (no shard_spec() access).
        const bool in1_is_nd_dram = tensor_b_memory_layout == TensorMemoryLayout::ND_SHARDED &&
                                    input_tensor_b.memory_config().buffer_type() == tt_metal::BufferType::DRAM;
        if (!(tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
              tensor_b_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || in1_is_nd_dram)) {
            return fmt::format(
                "{}: Input B memory layout must be WIDTH_SHARDED, HEIGHT_SHARDED, or DRAM ND_SHARDED, got: {}",
                config_name,
                tensor_b_memory_layout);
        }
        if (tensor_b_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            // Height-sharded in1 is only supported for DRAM batched matmuls
            if (!(input_tensor_b.memory_config().buffer_type() == tt_metal::BufferType::DRAM)) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input B is only supported in DRAM, got: {}",
                    config_name,
                    input_tensor_b.memory_config().buffer_type());
            }
            if (program_config.fuse_batch) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input B requires fuse_batch=false for batched matmul", config_name);
            }
            // Each DRAM bank must hold complete [K, N] matrices stacked vertically
            // K is the contracted dim: last dim of A, second-to-last of B
            const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
            const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
            const auto& in1_shard_spec = input_tensor_b.memory_config().shard_spec().value();
            uint32_t in1_shard_height_in_tiles = in1_shard_spec.shape[0] / in1_tile.get_height();
            uint32_t in1_shard_width_in_tiles = in1_shard_spec.shape[1] / in1_tile.get_width();
            uint32_t num_banks = in1_shard_spec.grid.num_cores();
            if (in1_shard_width_in_tiles != N) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input B shard width ({} tiles) must equal N ({} tiles)",
                    config_name,
                    in1_shard_width_in_tiles,
                    N);
            }
            if (in1_shard_height_in_tiles < K) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input B shard height ({} tiles) must be >= K ({} tiles)",
                    config_name,
                    in1_shard_height_in_tiles,
                    K);
            }
            if (in1_shard_height_in_tiles % K != 0) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input B shard height ({} tiles) must be divisible by K ({} tiles) "
                    "so each bank holds complete [K, N] matrices",
                    config_name,
                    in1_shard_height_in_tiles,
                    K);
            }
            uint32_t batches_per_bank = in1_shard_height_in_tiles / K;
            uint32_t B = get_batch_size(b_shape_padded);
            // The kernel addresses batch b via: bank_id = b / batches_per_bank.
            // Total shard capacity (batches_per_bank * num_banks) must be >= B
            // to ensure all batches map to valid banks. It may exceed B when the
            // batch dimension is padded up to distribute evenly across banks.
            if (batches_per_bank * num_banks < B) {
                return fmt::format(
                    "{}: HEIGHT_SHARDED input B: batches_per_bank ({}) * num_banks ({}) = {} must be >= "
                    "batch size B ({})",
                    config_name,
                    batches_per_bank,
                    num_banks,
                    batches_per_bank * num_banks,
                    B);
            }
        }
        if (input_tensor_b.memory_config().buffer_type() != tt_metal::BufferType::DRAM) {
            const auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
            if (!((input_tensor_a.memory_config().is_sharded() &&
                   tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) ||
                  tensor_a_memory_layout == TensorMemoryLayout::INTERLEAVED)) {
                return fmt::format(
                    "{}: Error - non-DRAM width sharded input B requires input A to be interleaved or height "
                    "sharded, rather than {}",
                    config_name,
                    tensor_a_memory_layout);
            }
            if (program_config.per_core_N !=
                (input_tensor_b.memory_config().shard_spec().value().shape[1] / in1_tile.get_width())) {
                return fmt::format(
                    "{}: per_core_N ({}) must equal input tensor B shard shape[1] / in1_tile.get_width() ({})",
                    config_name,
                    program_config.per_core_N,
                    (input_tensor_b.memory_config().shard_spec().value().shape[1] / in1_tile.get_width()));
            }
        }
        if (tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            if (!(input_tensor_b.memory_config().shard_spec()->grid.bounding_box().start_coord.y ==
                  input_tensor_b.memory_config().shard_spec()->grid.bounding_box().end_coord.y)) {
                return fmt::format(
                    "{}: Width-sharded input tensor B grid bounding box must have equal start and end y "
                    "coordinates, got start: {} vs end: {}",
                    config_name,
                    input_tensor_b.memory_config().shard_spec()->grid.bounding_box().start_coord.y,
                    input_tensor_b.memory_config().shard_spec()->grid.bounding_box().end_coord.y);
            }
        }
    }

    if (attributes.output_mem_config.is_sharded()) {
        if (attributes.output_mem_config.memory_layout() != TensorMemoryLayout::BLOCK_SHARDED) {
            return fmt::format(
                "{}: Output memory layout must be BLOCK_SHARDED, got: {}",
                config_name,
                attributes.output_mem_config.memory_layout());
        }
        uint32_t per_core_N = program_config.per_core_N;

        if (auto error = validate_output_subblock_block_divides_per_core_n(
                config_name,
                program_config.out_subblock_w,
                program_config.out_subblock_h,
                program_config.out_block_w,
                program_config.out_block_h,
                per_core_N);
            !error.empty()) {
            return error;
        }

        const uint32_t B = program_config.fuse_batch ? 1u : get_batch_size(a_shape_padded);
        operations::matmul::utilities::validate_block_sharded_output_batch(
            true, B, program_config.per_core_M, per_core_N);
    }
    return {};
}

// Reuse config: non-multicast block reuse. Validates per_core_M/N divisibility vs M/N,
// sharded A/B/output layouts, grid/shard-shape agreement, and rejects batch broadcast.
// (The post-visit work-split check is also Reuse-only, wired at the dispatch.)
std::string validate_matmul_reuse_config(
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseProgramConfig& program_config) {
    const auto config_name = ttsl::get_type_name(program_config);
    const auto M = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const auto total_M = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
    const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
    const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, /*tile=*/std::nullopt);
    uint32_t per_core_M = program_config.per_core_M;
    uint32_t per_core_N = program_config.per_core_N;
    if (per_core_M > M) {
        if (per_core_M % M != 0) {
            return fmt::format(
                "{}: per_core_M, {}, must be a multiple of M, {} if "
                "per_core_M > M!",
                config_name,
                per_core_M,
                M);
        }
        if (total_M % per_core_M != 0) {
            return fmt::format(
                "{}: input a total height, {}, must be divisible by "
                "per_core_M, {}!",
                config_name,
                total_M,
                per_core_M);
        }
    } else {
        if (M % per_core_M != 0) {
            return fmt::format("{}: per_core_M, {}, must divide M, {}, if per_core_M < M!", config_name, per_core_M, M);
        }
    }
    if (N != per_core_N) {
        return fmt::format("{}: N ({}) must equal per_core_N ({})", config_name, N, per_core_N);
    }
    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            return fmt::format(
                "{}: input A memory layout must not be WIDTH_SHARDED, got: {}",
                config_name,
                input_tensor_a.memory_config().memory_layout());
        }
        auto in0_shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;

        if (K != in0_shard_shape[1]) {
            return fmt::format("{}: K ({}) must equal in0 shard_shape[1] ({})", config_name, K, in0_shard_shape[1]);
        }
        if (in0_shard_shape[1] != program_config.in0_block_w * in0_tile.get_width()) {
            return fmt::format(
                "{}: in0 shard_shape[1] ({}) must equal in0_block_w ({}) * in0_tile width ({})",
                config_name,
                in0_shard_shape[1],
                program_config.in0_block_w,
                in0_tile.get_width());
        }
        if (per_core_M * in0_tile.get_height() != in0_shard_shape[0]) {
            return fmt::format(
                "{}: per_core_M ({}) * in0_tile height ({}) must equal in0 shard_shape[0] ({})",
                config_name,
                per_core_M,
                in0_tile.get_height(),
                in0_shard_shape[0]);
        }

        if (input_tensor_b.memory_config().is_sharded()) {
            if (input_tensor_a.memory_config().buffer_type() != input_tensor_b.memory_config().buffer_type()) {
                return fmt::format(
                    "{}: Input tensors A and B must have matching buffer types, got A: {} vs B: {}",
                    config_name,
                    input_tensor_a.memory_config().buffer_type(),
                    input_tensor_b.memory_config().buffer_type());
            }
            if (input_tensor_a.memory_config().memory_layout() != input_tensor_b.memory_config().memory_layout()) {
                return fmt::format(
                    "{}: Input tensors A and B must have matching memory layouts, got A: {} vs B: {}",
                    config_name,
                    input_tensor_a.memory_config().memory_layout(),
                    input_tensor_b.memory_config().memory_layout());
            }
            if (input_tensor_a.memory_config().shard_spec().value().grid !=
                input_tensor_b.memory_config().shard_spec().value().grid) {
                return fmt::format(
                    "{}: input A and B must have matching shard grids, got A: {} vs B: {}",
                    config_name,
                    input_tensor_a.memory_config().shard_spec().value().grid,
                    input_tensor_b.memory_config().shard_spec().value().grid);
            }
            if (input_tensor_a.memory_config().shard_spec().value().orientation !=
                input_tensor_b.memory_config().shard_spec().value().orientation) {
                return fmt::format(
                    "{}: Input tensors A and B must have matching shard orientations, got A: {} vs B: {}",
                    config_name,
                    input_tensor_a.memory_config().shard_spec().value().orientation,
                    input_tensor_b.memory_config().shard_spec().value().orientation);
            }
        }
        if (attributes.output_mem_config.is_sharded()) {
            if (auto error =
                    validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
                !error.empty()) {
                return error;
            }
        }
    }

    const auto batch_size_a = get_batch_size(a_shape_padded);
    const auto batch_size_b = get_batch_size(b_shape_padded);
    bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
    if (broadcast_batch) {
        return fmt::format("{}: Batch broadcasting is not supported for the chosen program config", config_name);
    }
    if (batch_size_a > 1 && batch_size_b > 1) {
        if (M % program_config.out_subblock_h != 0) {
            return fmt::format(
                "{}: out_subblock_h ({}) needs to divide M ({}) evenly and does not. "
                "Please update your program config.",
                config_name,
                program_config.out_subblock_h,
                M);
        }
    }

    if (input_tensor_b.memory_config().is_sharded()) {
        if (per_core_M % M != 0) {
            return fmt::format(
                "{}: per_core_M ({}) must be a multiple of M ({}) when input B is sharded", config_name, per_core_M, M);
        }
        if (input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            return fmt::format(
                "{}: Input B memory layout must not be WIDTH_SHARDED, got: {}",
                config_name,
                input_tensor_b.memory_config().memory_layout());
        }
        auto in1_shard_shape = input_tensor_b.memory_config().shard_spec().value().shape;
        if (in1_shard_shape[1] != b_shape_padded[-1]) {
            return fmt::format(
                "{}: Input B shard shape[1] ({}) must equal padded shape[-1] ({})",
                config_name,
                in1_shard_shape[1],
                b_shape_padded[-1]);
        }
        if (per_core_N * in1_tile.get_width() != in1_shard_shape[1]) {
            return fmt::format(
                "{}: per_core_N * in1_tile.get_width() ({}) must equal in1_shard_shape[1] ({})",
                config_name,
                per_core_N * in1_tile.get_width(),
                in1_shard_shape[1]);
        }
        if (in1_shard_shape[0] % K != 0) {
            return fmt::format(
                "{}: Input B shard shape[0] ({}) must be divisible by K ({})", config_name, in1_shard_shape[0], K);
        }
    }
    if (attributes.output_mem_config.is_sharded()) {
        if (attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            return fmt::format(
                "{}: Output memory layout must not be WIDTH_SHARDED, got: {}",
                config_name,
                attributes.output_mem_config.memory_layout());
        }
        if (!(program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1)) {
            return fmt::format(
                "{}: Either out_subblock_w ({}) must equal per_core_N ({}) or out_subblock_h ({}) must be 1",
                config_name,
                program_config.out_subblock_w,
                per_core_N,
                program_config.out_subblock_h);
        }
    }
    return {};
}

// Mcast1D config: 1D multicast. Validates the mcast_in0 and gather_in0 paths, the
// width-sharded and height-sharded in0 paths, and the output layout/subblock rules for
// 1-row vs 1-column grids.
std::string validate_matmul_mcast1d_config(
    const DeviceDesc& device,
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const std::optional<TensorSpec>& optional_bias,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    using namespace tt;  // BufferType/div_up were unqualified in the original std::visit scope
    const auto config_name = ttsl::get_type_name(program_config);
    if ((program_config.mcast_in0 && program_config.gather_in0)) {
        return fmt::format(
            "{}: Matmul1D does not support mcast_in0 and gather_in0 at the "
            "same time.",
            config_name);
    }
    if (!(program_config.gather_in0 || !program_config.stream_in1)) {
        return fmt::format(
            "{}: stream_in1 is the gather_in0 ring-rotation mode and requires gather_in0=true", config_name);
    }

    if (attributes.global_cb.has_value() && !program_config.gather_in0) {
        if (!program_config.mcast_in0) {
            return fmt::format("{}: global_cb without gather_in0 is supported only for mcast_in0=true", config_name);
        }
        if (auto error = validate_dram_sender_global_cb_mcast_in0_geometry(
                attributes.global_cb.value(), input_tensor_b, in1_tile, program_config);
            !error.empty()) {
            return error;
        }
        if (!(program_config.fuse_batch || get_batch_size(a_shape_padded) == 1)) {
            return fmt::format(
                "{}: mcast_in0 global_cb requires one effective activation batch, but fuse_batch={} and "
                "activation batch size={}",
                config_name,
                program_config.fuse_batch,
                get_batch_size(a_shape_padded));
        }
    }

    // Gather in0 specific validation
    if (program_config.gather_in0) {
        if (program_config.num_global_cb_receivers <= 0) {
            return fmt::format("{}: Num global CB receivers must be greater than 0.", config_name);
        }
        if (input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
            return fmt::format(
                "{}: input A must be WIDTH_SHARDED for gather_in0, got: {}",
                config_name,
                input_tensor_a.memory_config().memory_layout());
        }
        if (!(input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
              (input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
               input_tensor_b.memory_config().buffer_type() == tt_metal::BufferType::DRAM) ||
              // Receiver-contiguous Tensor prefetcher: in1 is an NdShardSpec DRAM
              // weight (reported as ND_SHARDED) whose data is delivered via the
              // global CB receivers, not read directly per its DRAM layout. The
              // weight's own layout is irrelevant to the matmul in this case.
              (attributes.global_cb.has_value() &&
               input_tensor_b.memory_config().buffer_type() == tt_metal::BufferType::DRAM))) {
            return fmt::format(
                "{}: Input tensor B must be width sharded, DRAM interleaved, or a DRAM weight fed "
                "via a global circular buffer when using gather_in0.",
                config_name);
        }
        if (!attributes.global_cb.has_value() && input_tensor_b.memory_config().is_sharded()) {
            if (input_tensor_b.memory_config().buffer_type() == tt_metal::BufferType::L1) {
                if (input_tensor_a.memory_config().shard_spec().value().grid !=
                    input_tensor_b.memory_config().shard_spec().value().grid) {
                    return fmt::format(
                        "{}: input A and B must be sharded on the same cores for gather_in0, got A: {} vs B: {}",
                        config_name,
                        input_tensor_a.memory_config().shard_spec().value().grid,
                        input_tensor_b.memory_config().shard_spec().value().grid);
                }
            }
        }
        if (!attributes.output_mem_config.is_sharded()) {
            return fmt::format("{}: Output tensor must be sharded when using gather_in0.", config_name);
        }
        if (!attributes.output_mem_config.shard_spec().has_value()) {
            return fmt::format("{}: Output shard spec must be provided when using gather_in0.", config_name);
        }

        if (!input_tensor_b.memory_config().is_sharded()) {
            if (attributes.global_cb.has_value()) {
                return fmt::format(
                    "{}: Global CB is not supported for DRAM_INTERLEAVED in1 when using gather_in0.", config_name);
            }
            if (input_tensor_b.layout() != Layout::TILE) {
                return fmt::format(
                    "{}: Input tensor B must be TILE_LAYOUT when DRAM_INTERLEAVED when using gather_in0.", config_name);
            }
            if (input_tensor_a.memory_config().shard_spec().value().grid !=
                attributes.output_mem_config.shard_spec().value().grid) {
                return fmt::format(
                    "{}: Input tensor A and output tensor must be sharded on the same cores when using gather_in0 "
                    "and in1 is DRAM_INTERLEAVED.",
                    config_name);
            }
        }

        if (!attributes.global_cb.has_value()) {
            if (program_config.num_global_cb_receivers != 1) {
                return fmt::format(
                    "{}: Num global CB receivers must be 1 when global CB is not provided.", config_name);
            }
        }

        // Cross-check program_config against the in1 weight shape (silent-hang guards),
        // gated on the DRAM-sender path. The two DRAM-sender weight layouts have
        // different bank->ring conventions, so dispatch per in1 memory_layout():
        //   * WIDTH_SHARDED (K-row-major): each bank holds one wide (K, N/num_banks)
        //     shard feeding the contiguous ring positions [b*rpb, (b+1)*rpb).
        //   * ND_SHARDED (receiver-contiguous): an NdShardSpec weight with round-robin
        //     shard placement and a strided bank->ring mapping.
        // NdShardSpec reports memory_layout() == ND_SHARDED (see MemoryConfig(BufferType,
        // NdShardSpec)); the prefetcher manager and validator key on the same enum.
        if (attributes.global_cb.has_value() && input_tensor_a.memory_config().is_sharded() &&
            tt::tt_metal::experimental::sender_core_type(attributes.global_cb.value()) ==
                tt::tt_metal::experimental::SenderCoreType::Dram) {
            const auto in1_layout = input_tensor_b.memory_config().memory_layout();
            if (in1_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                if (auto error = validate_dram_sender_global_cb_gather_in0_geometry(
                        attributes.global_cb.value(), input_tensor_a, b_shape_padded, in1_tile, program_config);
                    !error.empty()) {
                    return error;
                }
            } else if (in1_layout == TensorMemoryLayout::ND_SHARDED) {
                if (auto error = validate_dram_sender_global_cb_gather_in0_geometry_recv_contig(
                        attributes.global_cb.value(), input_tensor_a, b_shape_padded, in1_tile, program_config);
                    !error.empty()) {
                    return error;
                }
            } else {
                return fmt::format(
                    "{}: gather_in0 matmul with a DRAM-sender global CB requires in1 to be WIDTH_SHARDED "
                    "(K-row-major) or ND_SHARDED (receiver-contiguous), but got {}.",
                    config_name,
                    in1_layout);
            }
        }

        if (optional_bias.has_value()) {
            return fmt::format("{}: Bias is not supported when using gather_in0.", config_name);
        }
    } else {
        const auto device_grid_1d = device.grid;
        if (auto error = check_tensor_in_grid(input_tensor_a, device_grid_1d); !error.empty()) {
            return error;
        }
        if (!attributes.global_cb.has_value()) {
            if (auto error = check_tensor_in_grid(input_tensor_b, device_grid_1d); !error.empty()) {
                return error;
            }
        }
    }
    if (program_config.mcast_in0 || program_config.gather_in0) {
        if (input_tensor_a.memory_config().is_sharded()) {
            if (!program_config.fuse_batch) {
                return fmt::format("{}: fuse_batch must be enabled when input A is sharded", config_name);
            }
            if (input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
                return fmt::format(
                    "{}: input A must be WIDTH_SHARDED when mcast_in0 or gather_in0 is set, got: {}",
                    config_name,
                    input_tensor_a.memory_config().memory_layout());
            }
            if (attributes.output_mem_config.is_sharded()) {
                if (auto error = validate_input_a_output_mem_config_match(
                        config_name, input_tensor_a, attributes.output_mem_config);
                    !error.empty()) {
                    return error;
                }
            }
            if (input_tensor_a.memory_config().shard_spec().value().orientation != ShardOrientation::ROW_MAJOR) {
                return fmt::format(
                    "{}: input A shard orientation must be ROW_MAJOR, got: {}",
                    config_name,
                    input_tensor_a.memory_config().shard_spec().value().orientation);
            }
            const auto M =
                operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
            const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
            uint32_t per_core_M = program_config.per_core_M;
            auto shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;

            // No padding
            if (M != per_core_M) {
                return fmt::format("{}: M ({}) must equal per_core_M ({})", config_name, M, per_core_M);
            }
            if (per_core_M != (shard_shape[0] / in0_tile.get_height())) {
                return fmt::format(
                    "{}: per_core_M ({}) must equal shard_shape[0] ({}) / in0_tile height ({})",
                    config_name,
                    per_core_M,
                    shard_shape[0],
                    in0_tile.get_height());
            }
            if (K % program_config.in0_block_w != 0) {
                return fmt::format(
                    "{}: K ({}) must be divisible by in0_block_w ({})", config_name, K, program_config.in0_block_w);
            }
            if (!program_config.gather_in0) {  // Padding allowed for gather_in0
                if ((shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w != 0) {
                    return fmt::format(
                        "{}: shard_shape[1] ({}) / in0_tile width ({}) must be divisible by in0_block_w ({})",
                        config_name,
                        shard_shape[1],
                        in0_tile.get_width(),
                        program_config.in0_block_w);
                }
            }
        }
        if (attributes.output_mem_config.is_sharded()) {
            // Allow BLOCK_SHARDED on 1-row grids (equivalent to WIDTH_SHARDED)
            bool is_width_sharded = attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
            bool is_block_sharded_1d_row = false;
            if (attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED &&
                attributes.output_mem_config.shard_spec().has_value()) {
                auto grid_bbox = attributes.output_mem_config.shard_spec()->grid.bounding_box();
                is_block_sharded_1d_row = (grid_bbox.end_coord.y == grid_bbox.start_coord.y);
            }
            if (!(is_width_sharded || is_block_sharded_1d_row)) {
                return fmt::format(
                    "{}: output memory layout must be WIDTH_SHARDED or BLOCK_SHARDED on a 1-row grid, got: {}",
                    config_name,
                    attributes.output_mem_config.memory_layout());
            }
            const auto M =
                operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
            uint32_t per_core_M = program_config.per_core_M;
            uint32_t per_core_N = program_config.per_core_N;

            // No padding
            if (M != per_core_M) {
                return fmt::format("{}: M ({}) must equal per_core_M ({})", config_name, M, per_core_M);
            }

            if (auto error = validate_output_subblock_block_divides_per_core_n(
                    config_name,
                    program_config.out_subblock_w,
                    program_config.out_subblock_h,
                    program_config.out_block_w,
                    program_config.out_block_h,
                    per_core_N);
                !error.empty()) {
                return error;
            }
        }
        if (input_tensor_b.memory_config().buffer_type() == tt_metal::BufferType::L1 &&
            input_tensor_b.memory_config().is_sharded()) {
            if (input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
                return fmt::format(
                    "{}: input B in L1 must be WIDTH_SHARDED, got: {}",
                    config_name,
                    input_tensor_b.memory_config().memory_layout());
            }
            if (program_config.per_core_N !=
                (input_tensor_b.memory_config().shard_spec().value().shape[1] / in1_tile.get_width())) {
                return fmt::format(
                    "{}: input B shard width in tiles ({}) must equal per_core_N ({})",
                    config_name,
                    input_tensor_b.memory_config().shard_spec().value().shape[1] / in1_tile.get_width(),
                    program_config.per_core_N);
            }
            if (optional_bias.has_value()) {
                if (input_tensor_b.memory_config().shard_spec().value().shape[1] !=
                    optional_bias.value().memory_config().shard_spec().value().shape[1]) {
                    return fmt::format(
                        "{}: bias shard width ({}) must match input B shard width ({})",
                        config_name,
                        optional_bias.value().memory_config().shard_spec().value().shape[1],
                        input_tensor_b.memory_config().shard_spec().value().shape[1]);
                }
            }
        }
    } else {
        if (input_tensor_a.memory_config().is_sharded()) {
            if (!program_config.fuse_batch) {
                return fmt::format("{}: fuse_batch must be enabled when input A is sharded", config_name);
            }
            if (input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
                return fmt::format(
                    "{}: input A must be HEIGHT_SHARDED, got: {}",
                    config_name,
                    input_tensor_a.memory_config().memory_layout());
            }
            if (attributes.output_mem_config.is_sharded()) {
                if (auto error = validate_input_a_output_mem_config_match(
                        config_name, input_tensor_a, attributes.output_mem_config);
                    !error.empty()) {
                    return error;
                }
            }
            if (input_tensor_a.memory_config().shard_spec().value().orientation != ShardOrientation::ROW_MAJOR) {
                return fmt::format(
                    "{}: input A shard orientation must be ROW_MAJOR, got: {}",
                    config_name,
                    input_tensor_a.memory_config().shard_spec().value().orientation);
            }
            const auto M =
                operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
            const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
            uint32_t per_core_M = program_config.per_core_M;
            auto shard_shape = input_tensor_a.memory_config().shard_spec().value().shape;
            if (div_up(M, per_core_M) > input_tensor_a.memory_config().shard_spec().value().grid.num_cores()) {
                return fmt::format(
                    "{}: number of M blocks ceil(M/per_core_M)={} must not exceed input A shard grid cores ({})",
                    config_name,
                    div_up(M, per_core_M),
                    input_tensor_a.memory_config().shard_spec().value().grid.num_cores());
            }
            if (per_core_M != (shard_shape[0] / in0_tile.get_height())) {
                return fmt::format(
                    "{}: per_core_M ({}) must equal shard_shape[0] ({}) / in0_tile height ({})",
                    config_name,
                    per_core_M,
                    shard_shape[0],
                    in0_tile.get_height());
            }
            if (K % program_config.in0_block_w != 0) {
                return fmt::format(
                    "{}: K ({}) must be divisible by in0_block_w ({})", config_name, K, program_config.in0_block_w);
            }
            if (K != (shard_shape[1] / in0_tile.get_width())) {
                return fmt::format(
                    "{}: K ({}) must equal shard_shape[1] ({}) / in0_tile width ({})",
                    config_name,
                    K,
                    shard_shape[1],
                    in0_tile.get_width());
            }
        }
        if (attributes.output_mem_config.is_sharded()) {
            // Allow BLOCK_SHARDED on 1-column grids (equivalent to HEIGHT_SHARDED)
            bool is_height_sharded = attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED;
            bool is_block_sharded_1d_column = false;
            if (attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED &&
                attributes.output_mem_config.shard_spec().has_value()) {
                auto grid_bbox = attributes.output_mem_config.shard_spec()->grid.bounding_box();
                is_block_sharded_1d_column = (grid_bbox.end_coord.x == grid_bbox.start_coord.x);
            }
            if (!(is_height_sharded || is_block_sharded_1d_column)) {
                return fmt::format(
                    "{}: output memory layout must be HEIGHT_SHARDED or BLOCK_SHARDED on a 1-column grid, got: {}",
                    config_name,
                    attributes.output_mem_config.memory_layout());
            }
            const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
            uint32_t per_core_N = program_config.per_core_N;

            if (N != per_core_N) {
                return fmt::format("{}: N ({}) must equal per_core_N ({})", config_name, N, per_core_N);
            }
            if (auto error = validate_output_subblock_block_divides_per_core_n(
                    config_name,
                    program_config.out_subblock_w,
                    program_config.out_subblock_h,
                    program_config.out_block_w,
                    program_config.out_block_h,
                    per_core_N);
                !error.empty()) {
                return error;
            }
        }
        if (input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED) {
            return fmt::format(
                "{}: input B must be INTERLEAVED, got: {}",
                config_name,
                input_tensor_b.memory_config().memory_layout());
        }
    }
    return {};
}

// Shared Mcast2D/Mcast1D preamble — the fuse_batch gate. Fused batch requires B batch==1,
// and transpose_a is unsupported when real batch dims survive with M_per_batch > 1.
std::string validate_matmul_mcast_fuse_batch_preamble(
    std::string_view config_name,
    bool fuse_batch,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile) {
    if (fuse_batch) {
        if (get_batch_size(b_shape_padded) != 1) {
            return fmt::format(
                "{}: Matmul with fused batch requires input tensors of shapes BCMK*11KN=BCMN "
                "or equivalent. Please change the second input tensor or adjust the program config.",
                config_name);
        }
        if (attributes.transpose_a) {
            uint32_t batch_size_a = get_batch_size(a_shape_padded);
            uint32_t M_per_batch = a_shape_padded[-2] / in0_tile.get_height();
            if (!(batch_size_a == 1 || M_per_batch == 1)) {
                return fmt::format(
                    "{}: transpose_a with fuse_batch is not supported when batch dimensions "
                    "exist and M_per_batch > 1 (batch_size={}, M_per_batch={}, a_shape_padded={})",
                    config_name,
                    batch_size_a,
                    M_per_batch,
                    a_shape_padded);
            }
        }
    }
    return {};
}

std::string validate_matmul_reuse_work_split(
    const TensorSpec& input_tensor_a,
    const TensorSpec& input_tensor_b,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseProgramConfig& program_config,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::CoreRangeSet>& core_range_set) {
    const uint32_t B = ttnn::get_batch_size(a_shape_padded);
    const uint32_t Mt = get_M_dim(a_shape_padded, in0_tile, false);
    const uint32_t Nt = get_N_dim(b_shape_padded, in1_tile);
    const uint32_t per_core_M = program_config.per_core_M;
    const uint32_t per_core_N = program_config.per_core_N;
    if ((B * Mt) % per_core_M != 0) {
        return fmt::format("B * Mt ({}) must be divisible by per_core_M ({})", B * Mt, per_core_M);
    }
    if (Nt % per_core_N != 0) {
        return fmt::format("Nt ({}) must be divisible by per_core_N ({})", Nt, per_core_N);
    }
    const uint32_t num_output_blocks_total = (B * Mt / per_core_M) * (Nt / per_core_N);
    if (num_output_blocks_total <= 0) {
        return fmt::format(
            "matmul reuse produced zero output blocks (B={}, Mt={}, Nt={}, per_core_M={}, per_core_N={})",
            B,
            Mt,
            Nt,
            per_core_M,
            per_core_N);
    }

    std::optional<tt::tt_metal::ShardSpec> shard_spec = std::nullopt;
    if (input_tensor_a.memory_config().is_sharded()) {
        shard_spec = input_tensor_a.memory_config().shard_spec().value();
    } else if (input_tensor_b.memory_config().is_sharded()) {
        shard_spec = input_tensor_b.memory_config().shard_spec().value();
    } else if (
        output_mem_config.is_sharded() && output_mem_config.buffer_type() != tt::tt_metal::BufferType::DRAM &&
        output_mem_config.shard_spec().has_value()) {
        shard_spec = output_mem_config.shard_spec().value();
    }

    uint32_t num_cores = 0;
    if (shard_spec.has_value()) {
        num_cores = shard_spec->grid.num_cores();
    } else if (core_range_set.has_value()) {
        std::tie(num_cores, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
            tt::tt_metal::split_work_to_cores(core_range_set.value(), num_output_blocks_total);
    } else {
        const tt::tt_metal::CoreCoord grid = program_config.compute_with_storage_grid_size;
        std::tie(num_cores, std::ignore, std::ignore, std::ignore, std::ignore, std::ignore) =
            tt::tt_metal::split_work_to_cores(grid, num_output_blocks_total);
    }

    if (num_cores <= 0) {
        return fmt::format(
            "matmul reuse requires at least one active core, got 0 (num_output_blocks_total={})",
            num_output_blocks_total);
    }
    const uint32_t num_evenly_divided_output_blocks = num_output_blocks_total / num_cores;
    if (num_evenly_divided_output_blocks <= 0) {
        return fmt::format(
            "num_output_blocks_total ({}) must be >= num_cores ({}); some cores would have no work",
            num_output_blocks_total,
            num_cores);
    }
    return {};
}
}  // namespace

MatmulSpecs matmul_specs(
    const std::vector<Tensor>& input_tensors, const std::optional<const Tensor>& bias, const MatmulParams& attributes) {
    MatmulSpecs specs;
    for (const auto& tensor : input_tensors) {
        specs.inputs.push_back(tensor.tensor_spec());
    }
    if (bias.has_value()) {
        specs.bias = bias->tensor_spec();
    }
    specs.attributes = attributes;
    auto* device = input_tensors.at(0).device();
    specs.device.arch = device->arch();
    specs.device.grid = device->compute_with_storage_grid_size();
    specs.device.has_sub_devices = !device->get_sub_device_ids().empty();
    if (attributes.sub_device_id.has_value()) {
        specs.device.sub_device_workers =
            device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, attributes.sub_device_id.value());
    }
    return specs;
}

std::string program_config_error(
    const MatmulSpecs& specs, const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto& attributes = specs.attributes;
    const auto& device = specs.device;
    const auto& input_tensors = specs.inputs;
    const auto& input_tensor_a = specs.a();
    const auto& input_tensor_b = specs.b();
    const auto& optional_bias = specs.bias;
    const auto a_shape =
        operations::matmul::utilities::get_matmul_tensor_logical_shape(input_tensor_a, attributes.transpose_a);
    const auto b_shape =
        operations::matmul::utilities::get_matmul_tensor_logical_shape(input_tensor_b, attributes.transpose_b);
    const auto a_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_a, attributes.transpose_a);
    const auto b_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_b, attributes.transpose_b);
    const auto in0_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_a, attributes.transpose_a);
    const auto in1_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_b, attributes.transpose_b);

    // Checks shared by several configs (each self-filters by config)
    if (auto error = validate_matmul_tiny_tile_constraints(input_tensor_b, in0_tile, in1_tile, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_compute_grid_and_per_core_dims(device, chosen_program_config); !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_block_and_subblock_configuration(
            attributes, a_shape_padded, in0_tile, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_sharded_operand_grids_within_program_compute_grid(
            device, input_tensor_a, input_tensor_b, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_reuse_sharded_output_block_divisibility(
            input_tensor_a, input_tensor_b, a_shape_padded, b_shape_padded, in0_tile, in1_tile, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_work_distribution_and_gather_ring_topology(
            device,
            input_tensor_a,
            input_tensor_b,
            a_shape_padded,
            b_shape_padded,
            in0_tile,
            in1_tile,
            attributes.transpose_a,
            attributes.transpose_b,
            attributes.output_mem_config,
            chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_batch_compatibility(
            attributes, input_tensor_a, input_tensor_b, a_shape, b_shape, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_mcast1d_subdevice_worker_grid(device, attributes, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_input_count(attributes, input_tensors, input_tensor_b, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_bias_shape(
            optional_bias, in0_tile, in1_tile, a_shape_padded, b_shape, b_shape_padded, chosen_program_config);
        !error.empty()) {
        return error;
    }
    if (auto error = validate_matmul_untilize_out(attributes, chosen_program_config); !error.empty()) {
        return error;
    }

    // Per-config checks: the shared Mcast2D/Mcast1D fuse_batch preamble, then the config's own
    return std::visit(
        [&](const auto& program_config) -> std::string {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                if (auto error = validate_matmul_mcast_fuse_batch_preamble(
                        ttsl::get_type_name(program_config),
                        program_config.fuse_batch,
                        attributes,
                        a_shape_padded,
                        b_shape_padded,
                        in0_tile);
                    !error.empty()) {
                    return error;
                }
            }
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return validate_matmul_mcast1d_config(
                    device,
                    input_tensor_a,
                    input_tensor_b,
                    optional_bias,
                    attributes,
                    a_shape_padded,
                    b_shape_padded,
                    in0_tile,
                    in1_tile,
                    program_config);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return validate_matmul_dram_sharded_config(
                    input_tensor_a, input_tensor_b, attributes, a_shape_padded, in0_tile, program_config);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::
                                         MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                return validate_matmul_batched_dram_sharded_config(
                    input_tensor_a, input_tensor_b, attributes, a_shape_padded, in0_tile, program_config);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return validate_matmul_mcast2d_config(
                    device,
                    input_tensor_a,
                    input_tensor_b,
                    attributes,
                    a_shape_padded,
                    b_shape_padded,
                    in0_tile,
                    in1_tile,
                    program_config);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                if (auto error = validate_matmul_reuse_config(
                        input_tensor_a,
                        input_tensor_b,
                        attributes,
                        a_shape_padded,
                        b_shape_padded,
                        in0_tile,
                        in1_tile,
                        program_config);
                    !error.empty()) {
                    return error;
                }
                return validate_matmul_reuse_work_split(
                    input_tensor_a,
                    input_tensor_b,
                    a_shape_padded,
                    b_shape_padded,
                    in0_tile,
                    in1_tile,
                    program_config,
                    attributes.output_mem_config,
                    std::nullopt);
            } else {
                return validate_matmul_multicore_config(input_tensor_a, input_tensor_b, attributes, in0_tile, in1_tile);
            }
        },
        chosen_program_config);
}

}  // namespace ttnn::prim

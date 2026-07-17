// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

#include <string_view>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/experimental/global_circular_buffer.hpp"
#include "tt-metalium/work_split.hpp"
#include "tt_stl/reflection.hpp"
#include "tt_stl/unreachable.hpp"

namespace ttnn::prim {

namespace {

using tt::constants::TILE_HEIGHT;
using tt::constants::TILE_WIDTH;

// ===========================================================================
// VALIDATIONS FOR ALL CONFIGS: run for every program config, independent of which config
// is chosen.
// ===========================================================================

// Operand Basics: inputs must be on the same device, tilized,
// have a 32-wide inner tile dim (the hardware matmul works on 32x32 tiles), and both
// be floating-point.
void validate_matmul_operand_basics(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile) {
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");
    TT_FATAL(
        (input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE),
        "Inputs to matmul must be tilized");
    TT_FATAL(
        (in0_tile.get_width() == TILE_WIDTH && in1_tile.get_height() == TILE_WIDTH),
        "Matmul inner tile dim must be 32 (hardware constraint): got in0 tile width {}, in1 tile height {}",
        in0_tile.get_width(),
        in1_tile.get_height());
    TT_FATAL(
        is_floating_point(input_tensor_a.dtype()), "Unsupported data format for input A: {}", input_tensor_a.dtype());
    TT_FATAL(
        is_floating_point(input_tensor_b.dtype()), "Unsupported data format for input B: {}", input_tensor_b.dtype());
}

// Matrix Dimensions: checks ranks, K/M/N > 0, and that A's K equals B's K.
// The a_shape/b_shape passed in are already transpose-adjusted, so K sits at [-1] for A
// (width) and [-2] for B (height); B's K is read manually after b.rank >= 2 is verified.
void validate_matmul_matrix_dimensions(
    const ttnn::Shape& a_shape,
    const ttnn::Shape& b_shape,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile) {
    TT_FATAL(b_shape.rank() >= 2, "Matmul expects input B rank >= 2, got {}", b_shape.rank());
    TT_FATAL(a_shape[-1] > 0, "K dimension must be positive, got {}", a_shape[-1]);
    TT_FATAL(
        a_shape[-1] == b_shape[-2],
        "The width of the first tensor must be equal to the height of the second tensor. Mismatch: width={} height={}",
        a_shape[-1],
        b_shape[-2]);
    TT_FATAL(b_shape[-1] > 0, "Matmul requires N (columns of B) > 0, got {}", b_shape[-1]);
    if (a_shape.rank() >= 2) {
        TT_FATAL(a_shape[-2] > 0, "Matmul requires M (rows of A) > 0, got {}", a_shape[-2]);
    }
    const uint32_t Kt_a = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
    const uint32_t Kt_b = b_shape_padded[-2] / in1_tile.get_height();
    TT_FATAL(
        Kt_a > 0 && Kt_b > 0,
        "K dimension in tiles must be positive (input A: {} K-tiles, input B: {} K-tiles)",
        Kt_a,
        Kt_b);
    TT_FATAL(Kt_a == Kt_b, "K dimension in tiles must match between input A ({}) and input B ({})", Kt_a, Kt_b);
}

// Bfloat4 Tile Size: checks that a bfloat4 input has each tile dim >= 4. Only A's height
// and B's width are checked; the other two dims (A's width, B's height) are the K axis,
// already forced to 32 by the Operand Basics check above, so they can't be < 4.
void validate_matmul_bfloat4_tile_dims(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile) {
    constexpr uint32_t bfloat4_min_tile_height = 4;
    if (input_tensor_a.dtype() == DataType::BFLOAT4_B) {
        TT_FATAL(
            in0_tile.get_height() >= bfloat4_min_tile_height,
            "BFLOAT4_B matmul requires in0 tile height >= {} (got {})",
            bfloat4_min_tile_height,
            in0_tile.get_height());
    }
    if (input_tensor_b.dtype() == DataType::BFLOAT4_B) {
        TT_FATAL(
            in1_tile.get_width() >= bfloat4_min_tile_height,
            "BFLOAT4_B matmul requires in1 tile width >= {} (got {})",
            bfloat4_min_tile_height,
            in1_tile.get_width());
    }
}

// Tiny Tile Constraints: runs after the program config is chosen. Rejects tiny-tile
// combos that hang or deadlock on specific paths. See #42927.
void validate_matmul_tiny_tile_constraints(
    const Tensor& input_tensor_b,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const bool uses_tiny_outer_tile = (in0_tile.get_height() != TILE_HEIGHT || in1_tile.get_width() != TILE_WIDTH);
    if (!uses_tiny_outer_tile) {
        return;
    }

    // Transposed in1 with narrow tile width (16) is not supported by the LLK matmul
    // path: llk_math_matmul has no addr_mod handling for 32x16 transposed tiles.
    // Without this check the kernel hangs (CB producer/consumer deadlock). See #42927.
    const bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
    if (in1_transpose_tile && in1_tile.get_width() == 16) {
        TT_FATAL(
            false,
            "matmul does not support transposed in1 with tile width 16 (in1 tile is {}x{} with transpose). "
            "Use tile width 32 for transposed in1, or disable transpose_tile.",
            in1_tile.get_height(),
            in1_tile.get_width());
    }

    // Bfp compressed in1 dtypes (BFLOAT8_B, BFLOAT4_B) on the 2D/1D mcast paths hang for
    // tile_h < 16 — the LLK unpack/pack path for Bfp faces is not yet validated below
    // face_height 16 on these factories. The MatmulMultiCoreReuseProgramConfig path does
    // support smaller tile_h with Bfp dtypes, so this check is scoped to the mcast configs.
    // See #42927. This is a "not currently supported" throw, not a permanent fatal — it
    // should be removed when the underlying kernel limitation is resolved.
    const bool in1_is_bfp =
        (input_tensor_b.dtype() == DataType::BFLOAT8_B) || (input_tensor_b.dtype() == DataType::BFLOAT4_B);
    const bool is_mcast_config =
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>(chosen_program_config) ||
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config);
    if (in1_is_bfp && is_mcast_config && in0_tile.get_height() < 16) {
        TT_THROW(
            "matmul tiny-tile combo with in1 dtype {} and tile_h {} is not currently supported on the "
            "mcast program config path (requires tile_h >= 16); see issue #42927",
            input_tensor_b.dtype(),
            in0_tile.get_height());
    }
}

// Optional Tensors: checks at most one optional input (bias). A caller-provided output
// tensor must match the computed spec; otherwise the requested output mem-config must be
// compatible with it (1D block-sharded == HEIGHT/WIDTH sharded).
void validate_matmul_optional_tensors(
    const MatmulParams& attributes, const MatmulDeviceOperation::tensor_args_t& args) {
    const auto& optional_output_tensors = args.optional_output_tensors;
    const auto& optional_input_tensors = args.optional_input_tensors;
    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();

    TT_FATAL(
        optional_input_tensors.size() == 1,
        "Must have exactly 1 optional input tensor, got: {}",
        optional_input_tensors.size());

    const auto output_tensor_spec = MatmulDeviceOperation::compute_output_specs(attributes, args).at(0);
    if (is_optional_output_tensor) {
        const auto& optional_output_tensor_c = optional_output_tensors.at(0);
        const auto& optional_output_tensor_shape = optional_output_tensor_c->logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == output_tensor_spec.logical_shape(),
            "Shape of Optional Output Tensor {} doesn't match Output Tensor {}",
            optional_output_tensor_shape,
            output_tensor_spec.logical_shape());
        TT_FATAL(
            optional_output_tensor_c->dtype() == attributes.output_dtype.value(),
            "Type mismatch between optional output tensor {} & output tensor {}",
            optional_output_tensor_c->dtype(),
            attributes.output_dtype.value());
        TT_FATAL(
            optional_output_tensor_c->memory_config() == attributes.output_mem_config,
            "Memory config mismatch between optional output tensor {} & output "
            "tensor {}",
            optional_output_tensor_c->memory_config(),
            attributes.output_mem_config);
    } else {
        // A BLOCK_SHARDED request on a 1D grid is the same layout as HEIGHT/WIDTH sharded
        // (1-column grid = HEIGHT_SHARDED, 1-row grid = WIDTH_SHARDED). Work it out once here
        // and reuse it below for both the layout check and the warning.
        bool is_1d_column = false;
        bool is_1d_row = false;
        if (attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED &&
            attributes.output_mem_config.shard_spec().has_value()) {
            const auto grid_bbox = attributes.output_mem_config.shard_spec()->grid.bounding_box();
            is_1d_column = (grid_bbox.end_coord.x == grid_bbox.start_coord.x);
            is_1d_row = (grid_bbox.end_coord.y == grid_bbox.start_coord.y);
        }

        // Layouts must match, unless it is one of those equivalent 1D block conversions.
        bool memory_layout_compatible =
            output_tensor_spec.memory_config().memory_layout() == attributes.output_mem_config.memory_layout();
        if (!memory_layout_compatible) {
            memory_layout_compatible =
                (is_1d_column &&
                 output_tensor_spec.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) ||
                (is_1d_row && output_tensor_spec.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED);
        }
        TT_FATAL(
            memory_layout_compatible,
            "Mismatch between computed {} and provided {} mem config memory layout",
            output_tensor_spec.memory_config().memory_layout(),
            attributes.output_mem_config.memory_layout());
        TT_FATAL(
            output_tensor_spec.memory_config().buffer_type() == attributes.output_mem_config.buffer_type(),
            "Mismatch between computed {} and provided {} mem config buffer type",
            output_tensor_spec.memory_config().buffer_type(),
            attributes.output_mem_config.buffer_type());

        // A real 1D conversion (exactly one direction, not a single 1x1 core) is expected -
        // don't warn about it. Any other mismatch still warns.
        const bool is_single_core = is_1d_column && is_1d_row;
        const bool is_intentional_1d_conversion = !is_single_core && (is_1d_column || is_1d_row);
        if (attributes.output_mem_config.shard_spec().has_value() &&
            output_tensor_spec.memory_config() != attributes.output_mem_config && !is_intentional_1d_conversion) {
            log_warning(
                tt::LogOp,
                "Mismatch between computed {} and provided {} mem config. Using computed config.",
                output_tensor_spec.memory_config(),
                attributes.output_mem_config);
        }
    }
}

// Batch Compatibility: checks bcast -> B must be single-batch; non-bcast -> A and B must
// match rank + batch dims, unless A's batch is 1 and reused across B (the Mcast1D
// in0-reuse exception).
void validate_matmul_batch_compatibility(
    const MatmulParams& attributes,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const ttnn::Shape& a_shape,
    const ttnn::Shape& b_shape,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    TT_FATAL(attributes.bcast_batch.has_value(), "bcast_batch must be populated before matmul validation");
    if (attributes.bcast_batch.value()) {
        TT_FATAL(
            get_batch_size(b_shape) == 1,
            "Batch-broadcast matmul requires input B batch size 1 (shapes BCMK*11KN=BCMN), got B batch size {}",
            get_batch_size(b_shape));
    }

    // Validate batch dimensions for non-bcast matmul
    if (!attributes.bcast_batch.value()) {
        TT_FATAL(
            a_shape.rank() == b_shape.rank(),
            "Batched (non-bcast) matmul requires inputs of the same rank, got a_shape rank: {} vs b_shape rank: {}",
            a_shape.rank(),
            b_shape.rank());

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
            return !input_tensor_a.is_sharded() && !input_tensor_b.is_sharded();
        };

        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            TT_FATAL(
                a_shape[i] == b_shape[i] || (a_shape[i] == 1 && in0_reuse()),
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

// Input Count: checks there are normally exactly 2 inputs (activation + weight).
// Exception: the Mcast1D multi-tensor path (global_cb + DRAM-sharded weight) allows
// 1 activation + N weights, which must share shape/spec/layout/dtype.
void validate_matmul_input_count(
    const MatmulParams& attributes,
    const std::vector<Tensor>& input_tensors,
    const Tensor& input_tensor_b,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    if (std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            chosen_program_config) &&
        attributes.global_cb.has_value() && input_tensor_b.is_sharded() && input_tensor_b.buffer()->is_dram()) {
        for (uint32_t i = 1; i < input_tensors.size(); ++i) {
            TT_FATAL(
                input_tensor_b.logical_shape() == input_tensors[i].logical_shape(),
                "{}: for multi-tensor matmul, all weight tensors must have the same logical_shape, {} is not equal to "
                "{}",
                config_name,
                input_tensor_b.logical_shape(),
                input_tensors[i].logical_shape());
            TT_FATAL(
                input_tensor_b.padded_shape() == input_tensors[i].padded_shape(),
                "{}: for multi-tensor matmul, all weight tensors must have the same padded_shape {} is not equal to {}",
                config_name,
                input_tensor_b.padded_shape(),
                input_tensors[i].padded_shape());
            TT_FATAL(
                input_tensor_b.tensor_spec() == input_tensors[i].tensor_spec(),
                "{}: for multi-tensor matmul, all weight tensors must have the same tensor_spec {} is not equal to {}",
                config_name,
                input_tensor_b.tensor_spec(),
                input_tensors[i].tensor_spec());
            TT_FATAL(
                input_tensor_b.layout() == input_tensors[i].layout(),
                "{}: for multi-tensor matmul, all weight tensors must have the same layout {} is not equal to {}",
                config_name,
                input_tensor_b.layout(),
                input_tensors[i].layout());
            TT_FATAL(
                input_tensor_b.dtype() == input_tensors[i].dtype(),
                "{}: for multi-tensor matmul, all weight tensors must have the same dtype {} is not equal to {}",
                config_name,
                input_tensor_b.dtype(),
                input_tensors[i].dtype());
        }
    } else {
        TT_FATAL(
            input_tensors.size() == 2,
            "{}: Must have exactly 2 input tensors, got: {}",
            config_name,
            input_tensors.size());
    }
}

void validate_matmul_bias_shape(
    const std::optional<const Tensor>& optional_bias,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape,
    const ttnn::Shape& b_shape_padded,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (!optional_bias.has_value()) {
        return;
    }
    const auto& bias = optional_bias.value();
    auto bias_tile_shape = bias.tensor_spec().tile().get_tile_shape();
    TT_FATAL(
        (bias_tile_shape[0] == in0_tile.get_height() && bias_tile_shape[1] == in1_tile.get_width()),
        "Unsupported bias tile shape: bias tile ({}, {}) has to be (in0 tile height {}, in1 tile width {})",
        bias_tile_shape[0],
        bias_tile_shape[1],
        in0_tile.get_height(),
        in1_tile.get_width());
    TT_FATAL(bias.layout() == Layout::TILE, "Unsupported bias layout: {}, has to be TILE", bias.layout());
    const auto& bias_shape = bias.logical_shape();
    const auto& bias_shape_padded = bias.padded_shape();
    uint32_t bias_batch_size = get_batch_size(bias_shape);
    TT_FATAL(bias_batch_size == 1, "Unsupported bias shape: batch size must be 1, got {}", bias_batch_size);
    // MatmulMultiCoreReuseProgramConfig fuses a full per-batch [M, N] bias block, so its height must
    // cover exactly M; every other config indexes a single bias tile-row.
    const bool is_reuse_config =
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseProgramConfig>(chosen_program_config);
    const uint32_t Mt = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const uint32_t expected_bias_height = (is_reuse_config ? Mt : 1) * in0_tile.get_height();
    TT_FATAL(
        bias_shape_padded[-2] == expected_bias_height,
        "Unsupported bias shape: padded second last dimension of bias, {}, not equal to expected bias height, "
        "{} (tile height {} x {} bias tile-row(s))",
        bias_shape_padded[-2],
        expected_bias_height,
        in0_tile.get_height(),
        is_reuse_config ? Mt : 1);
    TT_FATAL(
        bias_shape_padded[-1] == b_shape_padded[-1],
        "Unsupported bias shape: padded last dimension of bias, {}, not "
        "equal to second input's padded last "
        "dimension, {}.",
        bias_shape_padded[-1],
        b_shape_padded[-1]);
    TT_FATAL(
        bias_shape[-1] >= b_shape[-1],
        "Unsupported bias shape: last dimension of bias, {}, not equal to "
        "or greater than second input's last "
        "dimension, {}.",
        bias_shape[-1],
        b_shape[-1]);

    // Fused bias with a narrow in1 tile (width 16) and full-height in0 (32) is not supported
    // by the broadcast-row bias kernel path. Without this check the kernel hangs. See #42927.
    if (in0_tile.get_height() == TILE_HEIGHT && in1_tile.get_width() == 16) {
        const bool in1_transpose_tile = in1_tile.get_transpose_of_faces() && in1_tile.get_transpose_within_face();
        if (!in1_transpose_tile) {
            TT_FATAL(
                false,
                "matmul fused bias does not support 32x16 narrow in1 tile (in0 tile height={}, in1 tile width={}). "
                "Use tile width 32 when bias is fused, or apply bias as a post-process add.",
                in0_tile.get_height(),
                in1_tile.get_width());
        }
    }
}

// Untilize Output: untilize_out means "give me row-major output." Allowed only with an
// explicit BF16/FP32 output dtype on the Mcast1D config; this check runs for every
// config so it can reject untilize_out on every other config.
void validate_matmul_untilize_out(
    const MatmulParams& attributes, const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    if (!attributes.untilize_out) {
        return;
    }
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    TT_FATAL(
        attributes.output_dtype.has_value(),
        "{}: Output dtype must be specified when untilize_out is true",
        config_name);
    TT_FATAL(
        (attributes.output_dtype.value() == DataType::BFLOAT16) ||
            (attributes.output_dtype.value() == DataType::FLOAT32),
        "{}: Unsupported data type: {}, only BFLOAT16 and FLOAT32 are supported",
        config_name,
        attributes.output_dtype.value());
    TT_FATAL(
        std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(chosen_program_config),
        "{}: untilize_out is not supported for this program config, only supported for "
        "MatmulMultiCoreReuseMultiCast1DProgramConfig",
        config_name);
}

// ===========================================================================
// VALIDATIONS FOR MULTIPLE PROGRAM CONFIGS: each function here runs for several (but not
// all) program configs that need the same checks, branching internally per config.
// Messages are config-named.
// ===========================================================================

// Block/Subblock Configuration: runs for Reuse, Mcast2D, Mcast1D (MultiCore, DRAMSharded,
// BatchedDRAMSharded skip it).
void validate_matmul_block_and_subblock_configuration(
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreProgramConfig> ||
                std::is_same_v<
                    ProgramConfigType,
                    operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig> ||
                std::is_same_v<
                    ProgramConfigType,
                    operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                return;
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                const uint32_t Kt = a_shape_padded[-1] / in0_tile.get_width();
                TT_FATAL(program_config.in0_block_w != 0, "{}: in0_block_w is 0, which is not valid", config_name);
                TT_FATAL(
                    Kt % program_config.in0_block_w == 0,
                    "{}: Kt ({}) must be divisible by in0_block_w ({})",
                    config_name,
                    Kt,
                    program_config.in0_block_w);
                TT_FATAL(
                    program_config.out_subblock_h != 0, "{}: out_subblock_h is 0, which is not valid", config_name);
                TT_FATAL(
                    program_config.out_subblock_w != 0, "{}: out_subblock_w is 0, which is not valid", config_name);
                if constexpr (
                    std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                    std::is_same_v<
                        ProgramConfigType,
                        operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    TT_FATAL(program_config.out_block_h != 0, "{}: out_block_h is 0, which is not valid", config_name);
                    TT_FATAL(program_config.out_block_w != 0, "{}: out_block_w is 0, which is not valid", config_name);
                    TT_FATAL(
                        program_config.out_block_h % program_config.out_subblock_h == 0,
                        "{}: out_block_h ({}) must be divisible by out_subblock_h ({})",
                        config_name,
                        program_config.out_block_h,
                        program_config.out_subblock_h);
                    TT_FATAL(
                        program_config.out_block_w % program_config.out_subblock_w == 0,
                        "{}: out_block_w ({}) must be divisible by out_subblock_w ({})",
                        config_name,
                        program_config.out_block_w,
                        program_config.out_subblock_w);
                    TT_FATAL(
                        program_config.per_core_M % program_config.out_block_h == 0,
                        "{}: per_core_M ({}) must be divisible by out_block_h ({})",
                        config_name,
                        program_config.per_core_M,
                        program_config.out_block_h);
                    TT_FATAL(
                        program_config.per_core_N % program_config.out_block_w == 0,
                        "{}: per_core_N ({}) must be divisible by out_block_w ({})",
                        config_name,
                        program_config.per_core_N,
                        program_config.out_block_w);
                }
                if constexpr (std::
                                  is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                    TT_FATAL(
                        program_config.per_core_M % program_config.out_subblock_h == 0,
                        "{}: per_core_M ({}) must be divisible by out_subblock_h ({})",
                        config_name,
                        program_config.per_core_M,
                        program_config.out_subblock_h);
                    TT_FATAL(
                        program_config.per_core_N % program_config.out_subblock_w == 0,
                        "{}: per_core_N ({}) must be divisible by out_subblock_w ({})",
                        config_name,
                        program_config.per_core_N,
                        program_config.out_subblock_w);
                }
                TT_FATAL(
                    attributes.compute_kernel_config.has_value(),
                    "{}: compute_kernel_config must be set for matmul subblock validation",
                    config_name);
                TT_FATAL(
                    attributes.output_tile.has_value(),
                    "{}: output_tile must be set for matmul subblock validation",
                    config_name);
                const uint32_t available_reg_count = ttnn::get_dest_reg_count(
                    attributes.compute_kernel_config.value(), attributes.output_tile.value().get_tile_shape());
                TT_FATAL(
                    program_config.out_subblock_h * program_config.out_subblock_w <= available_reg_count,
                    "{}: out_subblock_w {} times out_subblock_h {} needs to be at most {} to fit in hardware",
                    config_name,
                    program_config.out_subblock_w,
                    program_config.out_subblock_h,
                    available_reg_count);
            }
        },
        chosen_program_config);
}

// Helper: checks in0_block_w / per_core_M / per_core_N are non-zero.
void validate_matmul_nonzero_block_dims(
    std::string_view config_name, std::size_t in0_block_w, std::size_t per_core_M, std::size_t per_core_N) {
    TT_FATAL(in0_block_w != 0, "{}: in0_block_w is 0, which is not valid", config_name);
    TT_FATAL(per_core_M != 0, "{}: per_core_M is 0, which is not valid", config_name);
    TT_FATAL(per_core_N != 0, "{}: per_core_N is 0, which is not valid", config_name);
}

// Compute Grid & Per-Core Dims: skips MultiCore. For every other config it checks
// in0_block_w / per_core_M / per_core_N are non-zero (via the helper above); Reuse/
// Mcast2D/Mcast1D also check the program grid is non-zero and fits the device (Mcast1D
// gather_in0 skips that grid check — its grid comes from the input A shard grid).
void validate_matmul_compute_grid_and_per_core_dims(
    const Tensor& input_tensor_a, const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const CoreCoord device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreProgramConfig>) {
                return;  // MultiCore: no program grid / block dims to check
            } else {
                // Non-MultiCore. Grid-bounds check applies to Reuse/Mcast2D/Mcast1D only
                // (DRAMSharded/BatchedDRAMSharded map to DRAM banks).
                if constexpr (
                    std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig> ||
                    std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
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
                        TT_FATAL(
                            grid.x > 0 && grid.y > 0,
                            "{}: compute_with_storage_grid_size must be non-zero, got ({}, {})",
                            config_name,
                            grid.x,
                            grid.y);
                        TT_FATAL(
                            grid.x <= device_grid.x && grid.y <= device_grid.y,
                            "{}: compute_with_storage_grid_size ({}, {}) must fit within device grid ({}, {})",
                            config_name,
                            grid.x,
                            grid.y,
                            device_grid.x,
                            device_grid.y);
                    }
                }
                validate_matmul_nonzero_block_dims(
                    config_name, program_config.in0_block_w, program_config.per_core_M, program_config.per_core_N);
            }
        },
        chosen_program_config);
}

// Helper: an L1-sharded tensor's shard grid must fit within the given grid (DRAM skipped).
void check_tensor_in_grid(const Tensor& tensor, const CoreCoord& grid_size) {
    if (tensor.memory_config().is_sharded() && tensor.memory_config().buffer_type() != BufferType::DRAM) {
        const auto& shard_spec = tensor.memory_config().shard_spec().value();
        const auto& shard_grid = shard_spec.grid;
        TT_FATAL(
            grid_size.x > 0 && grid_size.y > 0,
            "compute grid size must be non-zero, got ({}, {})",
            grid_size.x,
            grid_size.y);
        const CoreRange range(CoreCoord(0, 0), CoreCoord(grid_size.x - 1, grid_size.y - 1));
        TT_FATAL(
            range.contains(shard_grid),
            "Tensor shard spec grid {} must lie within compute grid ({}, {})",
            shard_grid,
            grid_size.x,
            grid_size.y);
    }
}

// Helper: an L1-sharded output's shard grid must fit within the given extent (DRAM skipped).
void check_output_shard_grid_within_extent(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::CoreCoord& extent,
    std::string_view config_name) {
    if (!output_mem_config.is_sharded() || output_mem_config.buffer_type() == tt::tt_metal::BufferType::DRAM) {
        return;
    }
    if (!output_mem_config.shard_spec().has_value()) {
        return;
    }
    const auto& shard_grid = output_mem_config.shard_spec().value().grid;
    TT_FATAL(
        extent.x > 0 && extent.y > 0,
        "{}: device grid extent must be non-zero, got ({}, {})",
        config_name,
        extent.x,
        extent.y);
    const tt::tt_metal::CoreRange bbox(
        tt::tt_metal::CoreCoord(0, 0), tt::tt_metal::CoreCoord(extent.x - 1, extent.y - 1));
    TT_FATAL(
        bbox.contains(shard_grid),
        "{}: output shard grid {} must lie within extent {}",
        config_name,
        shard_grid,
        extent);
}

// Work Distribution & Gather Ring: runs for Reuse/Mcast2D/Mcast1D. Checks output blocks
// fit the cores; for Mcast1D gather_in0 also checks the ring setup (A sharded, sub-device
// present, hop cores not overlapping).
void validate_matmul_work_distribution_and_gather_ring_topology(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    bool transpose_a,
    bool transpose_b,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    const auto config_name = ttsl::get_active_type_name_in_variant(chosen_program_config);
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                const tt::tt_metal::CoreCoord device_extent = input_tensor_a.device()->compute_with_storage_grid_size();
                const auto& grid = program_config.compute_with_storage_grid_size;
                const auto Mt =
                    operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
                const auto Nt = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                const uint32_t per_core_M = program_config.per_core_M;
                const uint32_t per_core_N = program_config.per_core_N;
                const uint32_t num_cores = grid.x * grid.y;

                if (program_config.gather_in0) {
                    TT_FATAL(!transpose_a, "{}: transpose_a is not supported with gather_in0", config_name);
                    TT_FATAL(!transpose_b, "{}: transpose_b is not supported with gather_in0", config_name);
                    TT_FATAL(input_tensor_a.is_sharded(), "{}: gather_in0 requires input A to be sharded", config_name);
                    auto* device = input_tensor_a.device();
                    const auto& sub_device_ids = device->get_sub_device_ids();
                    TT_FATAL(
                        !sub_device_ids.empty(),
                        "{}: gather_in0 matmul requires at least one sub-device id on the device",
                        config_name);
                    if (!program_config.hop_cores.empty()) {
                        const tt::tt_metal::CoreRangeSet& worker_cores = input_tensor_a.shard_spec().value().grid;
                        TT_FATAL(
                            !program_config.hop_cores.intersects(worker_cores),
                            "{}: hop_cores must not overlap with input A shard grid. hop_cores={}, workers={}",
                            config_name,
                            program_config.hop_cores,
                            worker_cores);
                    }
                    check_output_shard_grid_within_extent(output_mem_config, device_extent, config_name);
                } else {
                    TT_FATAL(
                        program_config.hop_cores.empty(),
                        "{}: Hop cores are not supported for any mode besides gather_in0.",
                        config_name);
                    TT_FATAL(
                        Mt > 0 && Nt > 0,
                        "{}: Mt and Nt must be greater than zero in tiles (got Mt={}, Nt={})",
                        config_name,
                        Mt,
                        Nt);
                    const uint32_t num_blocks_y = ((Mt - 1) / per_core_M) + 1;
                    const uint32_t num_blocks_x = ((Nt - 1) / per_core_N) + 1;
                    const uint32_t num_blocks_total = num_blocks_y * num_blocks_x;
                    TT_FATAL(
                        num_blocks_total <= num_cores,
                        "{}: Number of blocks exceeds number of cores: {} blocks > {} cores",
                        config_name,
                        num_blocks_total,
                        num_cores);
                    if (program_config.mcast_in0) {
                        TT_FATAL(
                            num_blocks_y == 1,
                            "{}: mcast_in0 requires M ({}) to fit within a single per_core_M block ({}), got "
                            "num_blocks_y={}",
                            config_name,
                            Mt,
                            per_core_M,
                            num_blocks_y);
                    }
                    check_output_shard_grid_within_extent(output_mem_config, grid, config_name);
                }
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                const auto& grid = program_config.compute_with_storage_grid_size;
                const auto Mt =
                    operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
                const auto Nt = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                TT_FATAL(
                    Mt > 0 && Nt > 0,
                    "{}: Mt and Nt must be greater than zero in tiles (got Mt={}, Nt={})",
                    config_name,
                    Mt,
                    Nt);
                uint32_t num_blocks_y = ((Mt - 1) / program_config.per_core_M) + 1;
                uint32_t num_blocks_x = ((Nt - 1) / program_config.per_core_N) + 1;
                if (program_config.transpose_mcast) {
                    std::swap(num_blocks_x, num_blocks_y);
                }
                TT_FATAL(
                    num_blocks_x <= grid.x,
                    "{}: Num output blocks along x ({}) must be smaller than or equal to the number of columns in "
                    "compute grid ({})!",
                    config_name,
                    num_blocks_x,
                    grid.x);
                TT_FATAL(
                    num_blocks_y <= grid.y,
                    "{}: Num output blocks along y ({}) must be smaller than or equal to the number of rows in compute "
                    "grid ({})!",
                    config_name,
                    num_blocks_y,
                    grid.y);
                check_output_shard_grid_within_extent(output_mem_config, grid, config_name);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                // The factory selects all_cores from the first available shard spec: in0, then in1,
                // then output. Any of those can produce an offset grid (e.g. column 1 in a fused
                // chain). Use the device grid as the extent whenever any operand is sharded so we
                // don't incorrectly reject those grids against the origin-anchored config rect.
                const auto device_extent = input_tensor_a.device()->compute_with_storage_grid_size();
                const bool any_sharded = input_tensor_a.memory_config().is_sharded() ||
                                         input_tensor_b.memory_config().is_sharded() || output_mem_config.is_sharded();
                const auto effective_extent =
                    any_sharded ? device_extent : program_config.compute_with_storage_grid_size;
                check_output_shard_grid_within_extent(output_mem_config, effective_extent, config_name);
            } else {
                (void)transpose_a;
                (void)transpose_b;
            }
        },
        chosen_program_config);
}

// Sharded Operand Grids: Reuse config only. Each L1-sharded input's shard grid must fit
// within the device grid. Non-sharded inputs and DRAM inputs are not checked.
void validate_matmul_sharded_operand_grids_within_program_compute_grid(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                // When an input is sharded, the factory uses shard_spec.grid directly as all_cores
                // and ignores compute_with_storage_grid_size entirely. Validating the shard grid
                // against the origin-anchored compute_with_storage_grid_size rectangle incorrectly
                // rejects grids that don't start at (0,0) (e.g. column 1 in a multi-chain fused op).
                // The only physical constraint is that the shard grid fits within the device grid.
                // Non-sharded inputs are not checked here (the check only applies to L1-sharded tensors).
                const auto& config_grid = program_config.compute_with_storage_grid_size;
                const auto device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
                auto effective_grid_a = input_tensor_a.memory_config().is_sharded() ? device_grid : config_grid;
                auto effective_grid_b = input_tensor_b.memory_config().is_sharded() ? device_grid : config_grid;
                check_tensor_in_grid(input_tensor_a, effective_grid_a);
                check_tensor_in_grid(input_tensor_b, effective_grid_b);
            }
        },
        chosen_program_config);
}

// Output Block Divisibility: Reuse config only. If an input is sharded across N cores,
// the total number of output blocks must be a multiple of N so the work splits evenly
// across those cores; otherwise the program factory can't distribute it and fails.
void validate_matmul_reuse_sharded_output_block_divisibility(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulProgramConfig& chosen_program_config) {
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                // Mirror the shard_spec priority in MatmulMultiCoreReuseOptimizedProgramFactory::create_descriptor:
                // when in0 is L1-sharded its shard grid becomes the kernel grid; in1's grid is only consulted when
                // in0 is not sharded. num_output_blocks must divide evenly across that grid or the factory fatals.
                const Tensor* sharded = nullptr;
                if (input_tensor_a.is_sharded() && input_tensor_a.memory_config().buffer_type() != BufferType::DRAM) {
                    sharded = &input_tensor_a;
                } else if (
                    input_tensor_b.is_sharded() && input_tensor_b.memory_config().buffer_type() != BufferType::DRAM) {
                    sharded = &input_tensor_b;
                }
                if (sharded == nullptr) {
                    return;
                }
                const uint32_t B = get_batch_size(a_shape_padded);
                const uint32_t Mt = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, false);
                const uint32_t Nt = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                const uint32_t num_output_blocks =
                    (B * Mt / program_config.per_core_M) * (Nt / program_config.per_core_N);
                const uint32_t num_cores = sharded->shard_spec().value().grid.num_cores();
                TT_FATAL(
                    num_output_blocks % num_cores == 0,
                    "MatmulMultiCoreReuseProgramConfig: num_output_blocks ({}) must be evenly divisible by the "
                    "number of cores in the input shard grid ({})",
                    num_output_blocks,
                    num_cores);
            }
        },
        chosen_program_config);
}

// Helper: cross-validate a DRAM-sender global_cb's geometry against the matmul + weight shape.
// These catch silent-hang configs where the matmul reads more in1 pages than the prefetcher
// pushes (e.g. activation K padded past weight K). Gated by the caller on the DRAM-sender path
// because the worker-sender variant predates this work and uses different sizing/ordering
// conventions (no bank IDs; gcb_size = N * max_tile_size).
void validate_dram_sender_global_cb_gather_in0_geometry(
    const tt::tt_metal::experimental::GlobalCircularBuffer& gcb,
    const Tensor& input_tensor_a,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    const uint32_t ring_size = input_tensor_a.shard_spec().value().grid.num_cores();
    const uint32_t weight_K_tiles = b_shape_padded[-2] / in1_tile.get_height();
    const uint32_t weight_N_tiles = b_shape_padded[-1] / in1_tile.get_width();
    const uint32_t num_senders = gcb.sender_cores().num_cores();
    const uint32_t num_recv = gcb.receiver_cores().num_cores();
    const uint32_t recv_per_bank = static_cast<uint32_t>(program_config.num_global_cb_receivers);

    TT_FATAL(
        num_senders > 0 && num_recv == num_senders * recv_per_bank,
        "global_cb receiver count ({}) must equal num_senders ({}) * "
        "num_global_cb_receivers ({})",
        num_recv,
        num_senders,
        recv_per_bank);
    TT_FATAL(
        num_recv == ring_size,
        "global_cb receiver count ({}) must equal in0 (activation) ring_size "
        "({} = num_cores of in0.shard_spec.grid). Receivers and matmul workers "
        "must be the same set of cores.",
        num_recv,
        ring_size);

    // Semantic check: bank b must push to exactly the receivers at ring
    // positions [b*recv_per_bank, (b+1)*recv_per_bank). If satisfied, the
    // bank-to-receivers union also equals the activation grid as a set, so
    // we don't need a separate set-equality assertion (CoreRangeSet::operator==
    // compares ranges literally, which is brittle when one side is merged
    // into rectangles and the other is a flat list of single-core ranges).
    const auto& act_grid = input_tensor_a.shard_spec().value().grid;
    const auto ring_walk = tt::tt_metal::corerange_to_cores(act_grid, std::nullopt, /*row_wise=*/true);
    const auto& mapping = gcb.sender_receiver_core_mapping();
    TT_FATAL(
        mapping.size() * recv_per_bank == ring_walk.size(),
        "global_cb sender_receiver mapping ({} senders * {} receivers each) "
        "doesn't cover the matmul ring ({} cores)",
        mapping.size(),
        recv_per_bank,
        ring_walk.size());
    for (size_t bank_idx = 0; bank_idx < mapping.size(); ++bank_idx) {
        const auto bank_recvs =
            tt::tt_metal::corerange_to_cores(mapping[bank_idx].second, std::nullopt, /*row_wise=*/true);
        TT_FATAL(
            bank_recvs.size() == recv_per_bank,
            "Sender at bank index {} owns {} receivers; expected "
            "num_global_cb_receivers={}",
            bank_idx,
            bank_recvs.size(),
            recv_per_bank);
        for (size_t k = 0; k < recv_per_bank; ++k) {
            const size_t ring_pos = bank_idx * recv_per_bank + k;
            TT_FATAL(
                bank_recvs[k] == ring_walk[ring_pos],
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
    TT_FATAL(
        weight_K_tiles % ring_size == 0,
        "Weight K must be divisible by ring_size in tiles for gather_in0 + global_cb. "
        "Got weight_K_tiles={}, ring_size={} (remainder={}). The activation grid would "
        "pad K past the weight K, and the matmul would wait forever for in1 pages the "
        "prefetcher never pushes.",
        weight_K_tiles,
        ring_size,
        weight_K_tiles % ring_size);
    TT_FATAL(
        weight_N_tiles % num_senders == 0,
        "Weight N ({} tiles) must be divisible by num_senders ({}) so it shards "
        "evenly across the DRAM banks the global_cb senders cover",
        weight_N_tiles,
        num_senders);
    const uint32_t per_bank_N_tiles = weight_N_tiles / num_senders;
    TT_FATAL(
        per_bank_N_tiles % recv_per_bank == 0,
        "Weight per-bank N ({} tiles) must be divisible by num_global_cb_receivers ({})",
        per_bank_N_tiles,
        recv_per_bank);
    const uint32_t per_recv_N_tiles = per_bank_N_tiles / recv_per_bank;
    TT_FATAL(
        per_recv_N_tiles == program_config.per_core_N,
        "Matmul per_core_N ({}) must equal weight per-receiver N ({} = per_bank_N_tiles {} "
        "/ num_global_cb_receivers {})",
        program_config.per_core_N,
        per_recv_N_tiles,
        per_bank_N_tiles,
        recv_per_bank);
}

void validate_dram_sender_global_cb_gather_in0_geometry_recv_contig(
    const tt::tt_metal::experimental::GlobalCircularBuffer& gcb,
    const Tensor& input_tensor_a,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    const uint32_t ring_size = input_tensor_a.shard_spec().value().grid.num_cores();
    const uint32_t num_recv = gcb.receiver_cores().num_cores();
    TT_FATAL(
        num_recv == ring_size,
        "global_cb receiver count ({}) must equal in0 (activation) ring_size ({}). Receivers and matmul "
        "workers must be the same set of cores.",
        num_recv,
        ring_size);

    const uint32_t weight_K_tiles = b_shape_padded[-2] / in1_tile.get_height();
    const uint32_t weight_N_tiles = b_shape_padded[-1] / in1_tile.get_width();
    TT_FATAL(
        weight_K_tiles % ring_size == 0,
        "Weight K ({} tiles) must be divisible by ring_size ({}) for receiver-contiguous gather_in0 + "
        "global_cb (remainder {}). The activation grid pads K past the weight K and the matmul would "
        "wait forever for in1 pages the prefetcher never pushes.",
        weight_K_tiles,
        ring_size,
        weight_K_tiles % ring_size);
    TT_FATAL(
        weight_N_tiles % ring_size == 0,
        "Weight N ({} tiles) must be divisible by ring_size ({}) for receiver-contiguous gather_in0 + global_cb",
        weight_N_tiles,
        ring_size);
    const uint32_t per_recv_N_tiles = weight_N_tiles / ring_size;
    TT_FATAL(
        per_recv_N_tiles == program_config.per_core_N,
        "Matmul per_core_N ({}) must equal weight per-receiver N ({} = N_tiles {} / ring_size {}); otherwise "
        "the matmul's in1 page size disagrees with what the recv-contig prefetcher pushes.",
        program_config.per_core_N,
        per_recv_N_tiles,
        weight_N_tiles,
        ring_size);
}

// Helper: warns if a caller of MatmulDeviceOperation's static API hasn't populated
// allowed_worker_cores on a program_config variant that supports the field. ttnn::prim::matmul()
// normalizes its attributes before launch, but direct callers (e.g. CCL fused ops in
// ttnn/operations/experimental/ccl) need to invoke
// ttnn::operations::matmul::normalize_program_config() themselves. Downstream code in this file
// auto-populates via normalize_program_config on the chosen_program_config local, so this is
// currently advisory.
// TODO(#44529): convert this back to TT_FATAL once all callers have been updated.
void warn_if_allowed_worker_cores_missing(
    const std::optional<operations::matmul::MatmulProgramConfig>& program_config,
    [[maybe_unused]] std::string_view entry_point) {
    if (!program_config.has_value()) {
        return;
    }
    /* The following spammed CI logs too much; leave it in place to convert to TT_FATAL in the future.
    std::visit(
        [&](const auto& pc) {
            if constexpr (requires { pc.allowed_worker_cores; }) {
                if (!pc.allowed_worker_cores.has_value()) {
                    log_warning(
                        tt::LogOp,
                        "{}: program_config.allowed_worker_cores not populated on a MatmulProgramConfig variant "
                        "that supports the field. Auto-populating from compute_with_storage_grid_size. Callers "
                        "that bypass ttnn::prim::matmul() should invoke "
                        "ttnn::operations::matmul::normalize_program_config() on the program config first. "
                        "This will become a hard error in a future release.",
                        entry_point);
                }
            }
        },
        program_config.value());
        */
}

// Sub-Device Worker Grid: Mcast1D on a sub-device (non-gather). Checks the matmul grid
// fits on the sub-device's cores.
void validate_matmul_mcast1d_subdevice_worker_grid(
    const Tensor& input_tensor_a,
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
            auto* device = input_tensor_a.device();
            auto sub_device_cores =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, attributes.sub_device_id.value());
            auto bbox = sub_device_cores.bounding_box();
            TT_FATAL(
                sub_device_cores.num_cores() == bbox.size(),
                "{}: matmul_multicore_reuse_mcast_1d only supports rectangular sub-device worker grids. "
                "Got sub-device worker cores: {} (bounding box: {})",
                config_name,
                sub_device_cores,
                bbox);
            auto grid_size_1d = program_config_1d.allowed_worker_cores.value().bounding_box().grid_size();
            TT_FATAL(
                bbox.start_coord.x + grid_size_1d.x - 1 <= bbox.end_coord.x &&
                    bbox.start_coord.y + grid_size_1d.y - 1 <= bbox.end_coord.y,
                "{}: matmul grid_size {} anchored at sub-device start {} "
                "extends past the sub-device's worker bounding box {}",
                config_name,
                grid_size_1d,
                bbox.start_coord,
                bbox);
        }
    }
}

// Helper: input A and output must agree on buffer type + memory layout. Used by
// Reuse/Mcast2D/Mcast1D/DRAMSharded/BatchedDRAMSharded.
void validate_input_a_output_mem_config_match(
    std::string_view config_name, const Tensor& input_tensor_a, const tt::tt_metal::MemoryConfig& output_mem_config) {
    TT_FATAL(
        input_tensor_a.memory_config().buffer_type() == output_mem_config.buffer_type(),
        "{}: input A and output buffer types must match, got input: {} vs output: {}",
        config_name,
        input_tensor_a.memory_config().buffer_type(),
        output_mem_config.buffer_type());
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == output_mem_config.memory_layout(),
        "{}: input A and output memory layouts must match, got input: {} vs output: {}",
        config_name,
        input_tensor_a.memory_config().memory_layout(),
        output_mem_config.memory_layout());
}

// Helper: output subblock/block width must divide per_core_N. Used by
// Mcast2D/Mcast1D (Reuse keeps its own inline variant).
void validate_output_subblock_block_divides_per_core_n(
    std::string_view config_name,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t out_block_w,
    uint32_t out_block_h,
    uint32_t per_core_N) {
    TT_FATAL(
        out_subblock_w == per_core_N || out_subblock_h == 1,
        "{}: out_subblock_w ({}) must equal per_core_N ({}) or out_subblock_h ({}) must be 1",
        config_name,
        out_subblock_w,
        per_core_N,
        out_subblock_h);
    TT_FATAL(
        out_block_w == per_core_N || out_block_h == 1,
        "{}: out_block_w ({}) must equal per_core_N ({}) or out_block_h ({}) must be 1",
        config_name,
        out_block_w,
        per_core_N,
        out_block_h);
}

// ===========================================================================
// PROGRAM CONFIG SPECIFIC VALIDATIONS: one function per program config, holding the
// checks that fire for that config only. Dispatched from validate_on_program_cache_miss
// via one std::visit.
// ===========================================================================

// MultiCore config: the un-optimized fallback. Rejects tiny outer tiles and requires
// all operands + output to be INTERLEAVED.
void validate_matmul_multicore_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MatmulParams& attributes,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile) {
    constexpr auto config_name = ttsl::get_type_name<operations::matmul::MatmulMultiCoreProgramConfig>();
    const bool uses_tiny_outer_tile = (in0_tile.get_height() != TILE_HEIGHT || in1_tile.get_width() != TILE_WIDTH);
    if (uses_tiny_outer_tile) {
        TT_FATAL(
            false,
            "{}: matmul with non-optimized program config does not support tiny tile "
            "(in0 tile height={}, in1 tile width={}, expected TILE_HEIGHT={}, TILE_WIDTH={})",
            config_name,
            in0_tile.get_height(),
            in1_tile.get_width(),
            TILE_HEIGHT,
            TILE_WIDTH);
    }
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{}: Input A memory layout must be INTERLEAVED, got: {}",
        config_name,
        input_tensor_a.memory_config().memory_layout());
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{}: Input B memory layout must be INTERLEAVED, got: {}",
        config_name,
        input_tensor_b.memory_config().memory_layout());
    TT_FATAL(
        attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "{}: Output memory layout must be INTERLEAVED, got: {}",
        config_name,
        attributes.output_mem_config.memory_layout());
}

// DRAMSharded config: in0 width-sharded in L1, in1 width-sharded in DRAM; height must
// be a single tile (M == 1) and K/shard dims divide in0_block_w.
void validate_matmul_dram_sharded_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig& program_config) {
    const auto config_name = ttsl::get_type_name(program_config);

    // The DRAM-sharded mcast path is not yet validated for tile_h < 16 — it hangs for
    // all tile_h in {1,2,4,8} regardless of dtype or tile_w. Only tile_h in {16, 32} is
    // currently supported on this path. See #42927.
    if (in0_tile.get_height() < 16) {
        TT_THROW(
            "{}: matmul tiny-tile with tile_h {} is not currently supported on the DRAM-sharded mcast "
            "program config path (requires tile_h >= 16); see issue #42927",
            config_name,
            in0_tile.get_height());
    }

    TT_FATAL(
        input_tensor_a.is_sharded(), "{}: Input tensor A must be sharded for DRAM sharded program config", config_name);
    TT_FATAL(
        attributes.output_mem_config.is_sharded(),
        "{}: Output memory config must be sharded for DRAM sharded program config",
        config_name);
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "{}: Input A memory layout must be WIDTH_SHARDED, got: {}",
        config_name,
        input_tensor_a.memory_config().memory_layout());
    validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
    TT_FATAL(
        input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "{}: Input A shard orientation must be ROW_MAJOR, got: {}",
        config_name,
        input_tensor_a.shard_spec().value().orientation);
    const auto M = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
    const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
    uint32_t per_core_M = program_config.per_core_M;
    auto shard_shape = input_tensor_a.shard_spec().value().shape;

    // No padding
    TT_FATAL(M == per_core_M, "{}: M ({}) must equal per_core_M ({})", config_name, M, per_core_M);
    TT_FATAL(M == 1, "{}: currently only support in0 tensor height of tile height", config_name);
    TT_FATAL(
        per_core_M == (shard_shape[0] / in0_tile.get_height()),
        "{}: per_core_M ({}) must equal shard_shape[0] / in0_tile.get_height() ({})",
        config_name,
        per_core_M,
        (shard_shape[0] / in0_tile.get_height()));
    TT_FATAL(
        K % program_config.in0_block_w == 0,
        "{}: K ({}) must be divisible by in0_block_w ({})",
        config_name,
        K,
        program_config.in0_block_w);
    TT_FATAL(
        (shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
        "{}: shard_shape[1] / in0_tile.get_width() ({}) must be divisible by in0_block_w ({})",
        config_name,
        (shard_shape[1] / in0_tile.get_width()),
        program_config.in0_block_w);

    // tensor in1
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
        "{}: Input B memory layout must be WIDTH_SHARDED, got: {}",
        config_name,
        input_tensor_b.memory_config().memory_layout());
}

// BatchedDRAMSharded config: [1,B,M,K] x [1,B,K,N]: A height-sharded in L1, B height-
// sharded in DRAM, output height-sharded in L1; contracted dim divides in0_block_w.
void validate_matmul_batched_dram_sharded_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
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
    TT_FATAL(
        input_tensor_a.is_sharded(), "{}: Input tensor A must be sharded for batch-sharded DRAM matmul", config_name);
    TT_FATAL(
        attributes.output_mem_config.is_sharded(),
        "{}: Output memory config must be sharded for batch-sharded DRAM matmul",
        config_name);
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "{}: Input A memory layout must be HEIGHT_SHARDED for batch-sharded DRAM matmul, got: {}",
        config_name,
        input_tensor_a.memory_config().memory_layout());
    validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
    TT_FATAL(
        input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
        "{}: Input A shard orientation must be ROW_MAJOR, got: {}",
        config_name,
        input_tensor_a.shard_spec().value().orientation);

    // For batch sharding, the contracted dimension (N in A, N in B) must be divisible by in0_block_w
    const auto N_dim = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);  // K dim of A = N
    TT_FATAL(
        N_dim % program_config.in0_block_w == 0,
        "{}: N dimension ({}) must be divisible by in0_block_w ({})",
        config_name,
        N_dim,
        program_config.in0_block_w);

    // tensor in1: HEIGHT_SHARDED in DRAM (batch-sharded)
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
        "{}: Input B memory layout must be HEIGHT_SHARDED for batch-sharded DRAM matmul, got: {}",
        config_name,
        input_tensor_b.memory_config().memory_layout());
}

// Mcast2D config: block-sharded 2D multicast. Validates that sharded input A, input B,
// and the output have layouts, grids, and orientations consistent with a 2D multicast.
void validate_matmul_mcast2d_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& program_config) {
    using namespace tt;  // BufferType/div_up were unqualified in the original std::visit scope
    const auto config_name = ttsl::get_type_name(program_config);
    const tt::tt_metal::CoreCoord device_grid = input_tensor_a.device()->compute_with_storage_grid_size();
    check_tensor_in_grid(input_tensor_a, device_grid);
    check_tensor_in_grid(input_tensor_b, device_grid);
    if (input_tensor_a.memory_config().is_sharded()) {
        TT_FATAL(program_config.fuse_batch, "{}: Batch fusion is required when input A is sharded", config_name);
        auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
        const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
        uint32_t per_core_M = program_config.per_core_M;
        auto shard_shape = input_tensor_a.shard_spec().value().shape;

        TT_FATAL(
            tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "{}: Unsupported memory layout {}.",
            config_name,
            tensor_a_memory_layout);

        if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            if (program_config.transpose_mcast) {
                TT_FATAL(
                    input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR,
                    "{}: Input tensor A must have COL_MAJOR shard orientation for transpose MCAST, got: {}",
                    config_name,
                    input_tensor_a.shard_spec().value().orientation);
            } else {
                TT_FATAL(
                    input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                    "{}: Input tensor A must have ROW_MAJOR shard orientation for non-transpose MCAST, got: {}",
                    config_name,
                    input_tensor_a.shard_spec().value().orientation);
            }
            if (attributes.output_mem_config.is_sharded()) {
                validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
            }

        } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            TT_FATAL(
                !program_config.transpose_mcast,
                "{}: Transpose MCAST not supported with HEIGHT_SHARDED layout",
                config_name);
            TT_FATAL(
                K == program_config.in0_block_w,
                "{}: K ({}) must equal in0_block_w ({})",
                config_name,
                K,
                program_config.in0_block_w);
            TT_FATAL(
                program_config.in0_block_w == (shard_shape[1] / in0_tile.get_width()),
                "{}: in0_block_w ({}) must equal shard_shape[1] / in0_tile.get_width() ({})",
                config_name,
                program_config.in0_block_w,
                (shard_shape[1] / in0_tile.get_width()));
            TT_FATAL(
                input_tensor_a.shard_spec()->grid.bounding_box().start_coord.x ==
                    input_tensor_a.shard_spec()->grid.bounding_box().end_coord.x,
                "{}: HEIGHT_SHARDED input A must have a single-column shard grid (got x={} to x={}); use "
                "MatmulMultiCoreReuseProgramConfig for multi-column HEIGHT_SHARDED inputs",
                config_name,
                input_tensor_a.shard_spec()->grid.bounding_box().start_coord.x,
                input_tensor_a.shard_spec()->grid.bounding_box().end_coord.x);
        }

        TT_FATAL(
            per_core_M == (shard_shape[0] / in0_tile.get_height()),
            "{}: per_core_M ({}) must equal shard_shape[0] / in0_tile.get_height() ({})",
            config_name,
            per_core_M,
            (shard_shape[0] / in0_tile.get_height()));
        TT_FATAL(
            (shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
            "{}: shard_shape[1] / in0_tile.get_width() ({}) must be divisible by in0_block_w ({})",
            config_name,
            (shard_shape[1] / in0_tile.get_width()),
            program_config.in0_block_w);
    }

    if (input_tensor_b.memory_config().is_sharded()) {
        TT_FATAL(
            !program_config.transpose_mcast, "{}: Transpose MCAST not supported when input B is sharded", config_name);
        auto tensor_b_memory_layout = input_tensor_b.memory_config().memory_layout();
        // ND_SHARDED in1 in DRAM is read via the generic TensorAccessor path: the program
        // factory's in1_is_sharded only covers WIDTH/HEIGHT, so ND falls through to the
        // interleaved-style reader, which addresses the NdShardSpec layout from the accessor
        // args. The width/height-specific validation below is gated on those layouts, so ND
        // DRAM in1 skips it (no shard_spec() access).
        const bool in1_is_nd_dram = tensor_b_memory_layout == TensorMemoryLayout::ND_SHARDED &&
                                    input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::DRAM;
        TT_FATAL(
            tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED ||
                tensor_b_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED || in1_is_nd_dram,
            "{}: Input B memory layout must be WIDTH_SHARDED, HEIGHT_SHARDED, or DRAM ND_SHARDED, got: {}",
            config_name,
            tensor_b_memory_layout);
        if (tensor_b_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
            // Height-sharded in1 is only supported for DRAM batched matmuls
            TT_FATAL(
                input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::DRAM,
                "{}: HEIGHT_SHARDED input B is only supported in DRAM, got: {}",
                config_name,
                input_tensor_b.buffer()->buffer_type());
            TT_FATAL(
                !program_config.fuse_batch,
                "{}: HEIGHT_SHARDED input B requires fuse_batch=false for batched matmul",
                config_name);
            // Each DRAM bank must hold complete [K, N] matrices stacked vertically
            // K is the contracted dim: last dim of A, second-to-last of B
            const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
            const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
            const auto& in1_shard_spec = input_tensor_b.shard_spec().value();
            uint32_t in1_shard_height_in_tiles = in1_shard_spec.shape[0] / in1_tile.get_height();
            uint32_t in1_shard_width_in_tiles = in1_shard_spec.shape[1] / in1_tile.get_width();
            uint32_t num_banks = in1_shard_spec.grid.num_cores();
            TT_FATAL(
                in1_shard_width_in_tiles == N,
                "{}: HEIGHT_SHARDED input B shard width ({} tiles) must equal N ({} tiles)",
                config_name,
                in1_shard_width_in_tiles,
                N);
            TT_FATAL(
                in1_shard_height_in_tiles >= K,
                "{}: HEIGHT_SHARDED input B shard height ({} tiles) must be >= K ({} tiles)",
                config_name,
                in1_shard_height_in_tiles,
                K);
            TT_FATAL(
                in1_shard_height_in_tiles % K == 0,
                "{}: HEIGHT_SHARDED input B shard height ({} tiles) must be divisible by K ({} tiles) "
                "so each bank holds complete [K, N] matrices",
                config_name,
                in1_shard_height_in_tiles,
                K);
            uint32_t batches_per_bank = in1_shard_height_in_tiles / K;
            uint32_t B = get_batch_size(b_shape_padded);
            // The kernel addresses batch b via: bank_id = b / batches_per_bank.
            // Total shard capacity (batches_per_bank * num_banks) must be >= B
            // to ensure all batches map to valid banks. It may exceed B when the
            // batch dimension is padded up to distribute evenly across banks.
            TT_FATAL(
                batches_per_bank * num_banks >= B,
                "{}: HEIGHT_SHARDED input B: batches_per_bank ({}) * num_banks ({}) = {} must be >= "
                "batch size B ({})",
                config_name,
                batches_per_bank,
                num_banks,
                batches_per_bank * num_banks,
                B);
        }
        if (input_tensor_b.buffer()->buffer_type() != tt_metal::BufferType::DRAM) {
            const auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
            TT_FATAL(
                (input_tensor_a.memory_config().is_sharded() &&
                 tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) ||
                    tensor_a_memory_layout == TensorMemoryLayout::INTERLEAVED,
                "{}: Error - non-DRAM width sharded input B requires input A to be interleaved or height "
                "sharded, rather than {}",
                config_name,
                tensor_a_memory_layout);
            TT_FATAL(
                program_config.per_core_N == (input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width()),
                "{}: per_core_N ({}) must equal input tensor B shard shape[1] / in1_tile.get_width() ({})",
                config_name,
                program_config.per_core_N,
                (input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width()));
        }
        if (tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
            TT_FATAL(
                input_tensor_b.shard_spec()->grid.bounding_box().start_coord.y ==
                    input_tensor_b.shard_spec()->grid.bounding_box().end_coord.y,
                "{}: Width-sharded input tensor B grid bounding box must have equal start and end y "
                "coordinates, got start: {} vs end: {}",
                config_name,
                input_tensor_b.shard_spec()->grid.bounding_box().start_coord.y,
                input_tensor_b.shard_spec()->grid.bounding_box().end_coord.y);
        }
    }

    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(
            attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
            "{}: Output memory layout must be BLOCK_SHARDED, got: {}",
            config_name,
            attributes.output_mem_config.memory_layout());
        uint32_t per_core_N = program_config.per_core_N;

        validate_output_subblock_block_divides_per_core_n(
            config_name,
            program_config.out_subblock_w,
            program_config.out_subblock_h,
            program_config.out_block_w,
            program_config.out_block_h,
            per_core_N);
    }
}

// Reuse config: non-multicast block reuse. Validates per_core_M/N divisibility vs M/N,
// sharded A/B/output layouts, grid/shard-shape agreement, and rejects batch broadcast.
// (The post-visit work-split check is also Reuse-only, wired at the dispatch.)
void validate_matmul_reuse_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
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
        TT_FATAL(
            per_core_M % M == 0,
            "{}: per_core_M, {}, must be a multiple of M, {} if "
            "per_core_M > M!",
            config_name,
            per_core_M,
            M);
        TT_FATAL(
            total_M % per_core_M == 0,
            "{}: input a total height, {}, must be divisible by "
            "per_core_M, {}!",
            config_name,
            total_M,
            per_core_M);
    } else {
        TT_FATAL(
            M % per_core_M == 0,
            "{}: per_core_M, {}, must divide M, {}, if per_core_M < M!",
            config_name,
            per_core_M,
            M);
    }
    TT_FATAL(N == per_core_N, "{}: N ({}) must equal per_core_N ({})", config_name, N, per_core_N);
    if (input_tensor_a.is_sharded()) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "{}: input A memory layout must not be WIDTH_SHARDED, got: {}",
            config_name,
            input_tensor_a.memory_config().memory_layout());
        auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;

        TT_FATAL(
            K == in0_shard_shape[1],
            "{}: K ({}) must equal in0 shard_shape[1] ({})",
            config_name,
            K,
            in0_shard_shape[1]);
        TT_FATAL(
            in0_shard_shape[1] == program_config.in0_block_w * in0_tile.get_width(),
            "{}: in0 shard_shape[1] ({}) must equal in0_block_w ({}) * in0_tile width ({})",
            config_name,
            in0_shard_shape[1],
            program_config.in0_block_w,
            in0_tile.get_width());
        TT_FATAL(
            per_core_M * in0_tile.get_height() == in0_shard_shape[0],
            "{}: per_core_M ({}) * in0_tile height ({}) must equal in0 shard_shape[0] ({})",
            config_name,
            per_core_M,
            in0_tile.get_height(),
            in0_shard_shape[0]);

        if (input_tensor_b.is_sharded()) {
            TT_FATAL(
                input_tensor_a.memory_config().buffer_type() == input_tensor_b.memory_config().buffer_type(),
                "{}: Input tensors A and B must have matching buffer types, got A: {} vs B: {}",
                config_name,
                input_tensor_a.memory_config().buffer_type(),
                input_tensor_b.memory_config().buffer_type());
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == input_tensor_b.memory_config().memory_layout(),
                "{}: Input tensors A and B must have matching memory layouts, got A: {} vs B: {}",
                config_name,
                input_tensor_a.memory_config().memory_layout(),
                input_tensor_b.memory_config().memory_layout());
            TT_FATAL(
                input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                "{}: input A and B must have matching shard grids, got A: {} vs B: {}",
                config_name,
                input_tensor_a.shard_spec().value().grid,
                input_tensor_b.shard_spec().value().grid);
            TT_FATAL(
                input_tensor_a.shard_spec().value().orientation == input_tensor_b.shard_spec().value().orientation,
                "{}: Input tensors A and B must have matching shard orientations, got A: {} vs B: {}",
                config_name,
                input_tensor_a.shard_spec().value().orientation,
                input_tensor_b.shard_spec().value().orientation);
        }
        if (attributes.output_mem_config.is_sharded()) {
            validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
        }
    }

    const auto batch_size_a = get_batch_size(a_shape_padded);
    const auto batch_size_b = get_batch_size(b_shape_padded);
    bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
    TT_FATAL(!broadcast_batch, "{}: Batch broadcasting is not supported for the chosen program config", config_name);
    if (batch_size_a > 1 && batch_size_b > 1) {
        TT_FATAL(
            M % program_config.out_subblock_h == 0,
            "{}: out_subblock_h ({}) needs to divide M ({}) evenly and does not. "
            "Please update your program config.",
            config_name,
            program_config.out_subblock_h,
            M);
    }

    if (input_tensor_b.is_sharded()) {
        TT_FATAL(
            per_core_M % M == 0,
            "{}: per_core_M ({}) must be a multiple of M ({}) when input B is sharded",
            config_name,
            per_core_M,
            M);
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "{}: Input B memory layout must not be WIDTH_SHARDED, got: {}",
            config_name,
            input_tensor_b.memory_config().memory_layout());
        auto in1_shard_shape = input_tensor_b.shard_spec().value().shape;
        TT_FATAL(
            in1_shard_shape[1] == b_shape_padded[-1],
            "{}: Input B shard shape[1] ({}) must equal padded shape[-1] ({})",
            config_name,
            in1_shard_shape[1],
            b_shape_padded[-1]);
        TT_FATAL(
            per_core_N * in1_tile.get_width() == in1_shard_shape[1],
            "{}: per_core_N * in1_tile.get_width() ({}) must equal in1_shard_shape[1] ({})",
            config_name,
            per_core_N * in1_tile.get_width(),
            in1_shard_shape[1]);
        TT_FATAL(
            in1_shard_shape[0] % K == 0,
            "{}: Input B shard shape[0] ({}) must be divisible by K ({})",
            config_name,
            in1_shard_shape[0],
            K);
    }
    if (attributes.output_mem_config.is_sharded()) {
        TT_FATAL(
            attributes.output_mem_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
            "{}: Output memory layout must not be WIDTH_SHARDED, got: {}",
            config_name,
            attributes.output_mem_config.memory_layout());
        TT_FATAL(
            program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
            "{}: Either out_subblock_w ({}) must equal per_core_N ({}) or out_subblock_h ({}) must be 1",
            config_name,
            program_config.out_subblock_w,
            per_core_N,
            program_config.out_subblock_h);
    }
}

// Mcast1D config: 1D multicast. Validates the mcast_in0 and gather_in0 paths, the
// width-sharded and height-sharded in0 paths, and the output layout/subblock rules for
// 1-row vs 1-column grids.
void validate_matmul_mcast1d_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<const Tensor>& optional_bias,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig& program_config) {
    using namespace tt;  // BufferType/div_up were unqualified in the original std::visit scope
    const auto config_name = ttsl::get_type_name(program_config);
    TT_FATAL(
        !(program_config.mcast_in0 && program_config.gather_in0),
        "{}: Matmul1D does not support mcast_in0 and gather_in0 at the "
        "same time.",
        config_name);

    // Gather in0 specific validation
    if (program_config.gather_in0) {
        TT_FATAL(
            program_config.num_global_cb_receivers > 0,
            "{}: Num global CB receivers must be greater than 0.",
            config_name);
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
            "{}: input A must be WIDTH_SHARDED for gather_in0, got: {}",
            config_name,
            input_tensor_a.memory_config().memory_layout());
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                (input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                 input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::DRAM) ||
                // Receiver-contiguous Tensor prefetcher: in1 is an NdShardSpec DRAM
                // weight (reported as ND_SHARDED) whose data is delivered via the
                // global CB receivers, not read directly per its DRAM layout. The
                // weight's own layout is irrelevant to the matmul in this case.
                (attributes.global_cb.has_value() &&
                 input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::DRAM),
            "{}: Input tensor B must be width sharded, DRAM interleaved, or a DRAM weight fed "
            "via a global circular buffer when using gather_in0.",
            config_name);
        if (!attributes.global_cb.has_value() && input_tensor_b.is_sharded()) {
            if (input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::L1) {
                TT_FATAL(
                    input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                    "{}: input A and B must be sharded on the same cores for gather_in0, got A: {} vs B: {}",
                    config_name,
                    input_tensor_a.shard_spec().value().grid,
                    input_tensor_b.shard_spec().value().grid);
            }
        }
        TT_FATAL(
            attributes.output_mem_config.is_sharded(),
            "{}: Output tensor must be sharded when using gather_in0.",
            config_name);
        TT_FATAL(
            attributes.output_mem_config.shard_spec().has_value(),
            "{}: Output shard spec must be provided when using gather_in0.",
            config_name);

        if (!input_tensor_b.is_sharded()) {
            TT_FATAL(
                !attributes.global_cb.has_value(),
                "{}: Global CB is not supported for DRAM_INTERLEAVED in1 when using gather_in0.",
                config_name);
            TT_FATAL(
                input_tensor_b.layout() == Layout::TILE,
                "{}: Input tensor B must be TILE_LAYOUT when DRAM_INTERLEAVED when using gather_in0.",
                config_name);
            TT_FATAL(
                input_tensor_a.shard_spec().value().grid == attributes.output_mem_config.shard_spec().value().grid,
                "{}: Input tensor A and output tensor must be sharded on the same cores when using gather_in0 "
                "and in1 is DRAM_INTERLEAVED.",
                config_name);
        }

        if (!attributes.global_cb.has_value()) {
            TT_FATAL(
                program_config.num_global_cb_receivers == 1,
                "{}: Num global CB receivers must be 1 when global CB is not provided.",
                config_name);
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
        if (attributes.global_cb.has_value() && input_tensor_a.is_sharded() &&
            tt::tt_metal::experimental::sender_core_type(attributes.global_cb.value()) ==
                tt::tt_metal::experimental::SenderCoreType::Dram) {
            const auto in1_layout = input_tensor_b.memory_config().memory_layout();
            if (in1_layout == TensorMemoryLayout::WIDTH_SHARDED) {
                validate_dram_sender_global_cb_gather_in0_geometry(
                    attributes.global_cb.value(), input_tensor_a, b_shape_padded, in1_tile, program_config);
            } else if (in1_layout == TensorMemoryLayout::ND_SHARDED) {
                validate_dram_sender_global_cb_gather_in0_geometry_recv_contig(
                    attributes.global_cb.value(), input_tensor_a, b_shape_padded, in1_tile, program_config);
            } else {
                TT_FATAL(
                    false,
                    "{}: gather_in0 matmul with a DRAM-sender global CB requires in1 to be WIDTH_SHARDED "
                    "(K-row-major) or ND_SHARDED (receiver-contiguous), but got {}.",
                    config_name,
                    in1_layout);
            }
        }

        TT_FATAL(!optional_bias.has_value(), "{}: Bias is not supported when using gather_in0.", config_name);
    } else {
        const auto device_grid_1d = input_tensor_a.device()->compute_with_storage_grid_size();
        check_tensor_in_grid(input_tensor_a, device_grid_1d);
        check_tensor_in_grid(input_tensor_b, device_grid_1d);
    }
    if (program_config.mcast_in0 || program_config.gather_in0) {
        if (input_tensor_a.is_sharded()) {
            TT_FATAL(program_config.fuse_batch, "{}: fuse_batch must be enabled when input A is sharded", config_name);
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                "{}: input A must be WIDTH_SHARDED when mcast_in0 or gather_in0 is set, got: {}",
                config_name,
                input_tensor_a.memory_config().memory_layout());
            if (attributes.output_mem_config.is_sharded()) {
                validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
            }
            TT_FATAL(
                input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                "{}: input A shard orientation must be ROW_MAJOR, got: {}",
                config_name,
                input_tensor_a.shard_spec().value().orientation);
            const auto M =
                operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
            const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
            uint32_t per_core_M = program_config.per_core_M;
            auto shard_shape = input_tensor_a.shard_spec().value().shape;

            // No padding
            TT_FATAL(M == per_core_M, "{}: M ({}) must equal per_core_M ({})", config_name, M, per_core_M);
            TT_FATAL(
                per_core_M == (shard_shape[0] / in0_tile.get_height()),
                "{}: per_core_M ({}) must equal shard_shape[0] ({}) / in0_tile height ({})",
                config_name,
                per_core_M,
                shard_shape[0],
                in0_tile.get_height());
            TT_FATAL(
                K % program_config.in0_block_w == 0,
                "{}: K ({}) must be divisible by in0_block_w ({})",
                config_name,
                K,
                program_config.in0_block_w);
            if (!program_config.gather_in0) {  // Padding allowed for gather_in0
                TT_FATAL(
                    (shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
                    "{}: shard_shape[1] ({}) / in0_tile width ({}) must be divisible by in0_block_w ({})",
                    config_name,
                    shard_shape[1],
                    in0_tile.get_width(),
                    program_config.in0_block_w);
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
            TT_FATAL(
                is_width_sharded || is_block_sharded_1d_row,
                "{}: output memory layout must be WIDTH_SHARDED or BLOCK_SHARDED on a 1-row grid, got: {}",
                config_name,
                attributes.output_mem_config.memory_layout());
            const auto M =
                operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
            uint32_t per_core_M = program_config.per_core_M;
            uint32_t per_core_N = program_config.per_core_N;

            // No padding
            TT_FATAL(M == per_core_M, "{}: M ({}) must equal per_core_M ({})", config_name, M, per_core_M);

            validate_output_subblock_block_divides_per_core_n(
                config_name,
                program_config.out_subblock_w,
                program_config.out_subblock_h,
                program_config.out_block_w,
                program_config.out_block_h,
                per_core_N);
        }
        if (input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::L1 &&
            input_tensor_b.memory_config().is_sharded()) {
            TT_FATAL(
                input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                "{}: input B in L1 must be WIDTH_SHARDED, got: {}",
                config_name,
                input_tensor_b.memory_config().memory_layout());
            TT_FATAL(
                program_config.per_core_N == (input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width()),
                "{}: input B shard width in tiles ({}) must equal per_core_N ({})",
                config_name,
                input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width(),
                program_config.per_core_N);
            if (optional_bias.has_value()) {
                TT_FATAL(
                    input_tensor_b.shard_spec().value().shape[1] == optional_bias.value().shard_spec().value().shape[1],
                    "{}: bias shard width ({}) must match input B shard width ({})",
                    config_name,
                    optional_bias.value().shard_spec().value().shape[1],
                    input_tensor_b.shard_spec().value().shape[1]);
            }
        }
    } else {
        if (input_tensor_a.memory_config().is_sharded()) {
            TT_FATAL(program_config.fuse_batch, "{}: fuse_batch must be enabled when input A is sharded", config_name);
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "{}: input A must be HEIGHT_SHARDED, got: {}",
                config_name,
                input_tensor_a.memory_config().memory_layout());
            if (attributes.output_mem_config.is_sharded()) {
                validate_input_a_output_mem_config_match(config_name, input_tensor_a, attributes.output_mem_config);
            }
            TT_FATAL(
                input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                "{}: input A shard orientation must be ROW_MAJOR, got: {}",
                config_name,
                input_tensor_a.shard_spec().value().orientation);
            const auto M =
                operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
            const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
            uint32_t per_core_M = program_config.per_core_M;
            auto shard_shape = input_tensor_a.shard_spec().value().shape;
            TT_FATAL(
                div_up(M, per_core_M) <= input_tensor_a.shard_spec().value().grid.num_cores(),
                "{}: number of M blocks ceil(M/per_core_M)={} must not exceed input A shard grid cores ({})",
                config_name,
                div_up(M, per_core_M),
                input_tensor_a.shard_spec().value().grid.num_cores());
            TT_FATAL(
                per_core_M == (shard_shape[0] / in0_tile.get_height()),
                "{}: per_core_M ({}) must equal shard_shape[0] ({}) / in0_tile height ({})",
                config_name,
                per_core_M,
                shard_shape[0],
                in0_tile.get_height());
            TT_FATAL(
                K % program_config.in0_block_w == 0,
                "{}: K ({}) must be divisible by in0_block_w ({})",
                config_name,
                K,
                program_config.in0_block_w);
            TT_FATAL(
                K == (shard_shape[1] / in0_tile.get_width()),
                "{}: K ({}) must equal shard_shape[1] ({}) / in0_tile width ({})",
                config_name,
                K,
                shard_shape[1],
                in0_tile.get_width());
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
            TT_FATAL(
                is_height_sharded || is_block_sharded_1d_column,
                "{}: output memory layout must be HEIGHT_SHARDED or BLOCK_SHARDED on a 1-column grid, got: {}",
                config_name,
                attributes.output_mem_config.memory_layout());
            const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
            uint32_t per_core_N = program_config.per_core_N;

            TT_FATAL(N == per_core_N, "{}: N ({}) must equal per_core_N ({})", config_name, N, per_core_N);
            validate_output_subblock_block_divides_per_core_n(
                config_name,
                program_config.out_subblock_w,
                program_config.out_subblock_h,
                program_config.out_block_w,
                program_config.out_block_h,
                per_core_N);
        }
        TT_FATAL(
            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "{}: input B must be INTERLEAVED, got: {}",
            config_name,
            input_tensor_b.memory_config().memory_layout());
    }
}

// Shared Mcast2D/Mcast1D preamble — the fuse_batch gate. Fused batch requires B batch==1,
// and transpose_a is unsupported when real batch dims survive with M_per_batch > 1.
void validate_matmul_mcast_fuse_batch_preamble(
    std::string_view config_name,
    bool fuse_batch,
    const MatmulParams& attributes,
    const ttnn::Shape& a_shape_padded,
    const ttnn::Shape& b_shape_padded,
    const tt::tt_metal::Tile& in0_tile) {
    if (fuse_batch) {
        TT_FATAL(
            get_batch_size(b_shape_padded) == 1,
            "{}: Matmul with fused batch requires input tensors of shapes BCMK*11KN=BCMN "
            "or equivalent. Please change the second input tensor or adjust the program config.",
            config_name);
        if (attributes.transpose_a) {
            uint32_t batch_size_a = get_batch_size(a_shape_padded);
            uint32_t M_per_batch = a_shape_padded[-2] / in0_tile.get_height();
            TT_FATAL(
                batch_size_a == 1 || M_per_batch == 1,
                "{}: transpose_a with fuse_batch is not supported when batch dimensions "
                "exist and M_per_batch > 1 (batch_size={}, M_per_batch={}, a_shape_padded={})",
                config_name,
                batch_size_a,
                M_per_batch,
                a_shape_padded);
        }
    }
}

// Helper: returns whether batch broadcasting applies: true when input B has batch size 1 (B is
// reused across A's batches). Used by compute_output_specs, not by validate.
bool get_broadcast_batch(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const std::optional<const operations::matmul::MatmulProgramConfig>& matmul_program_config) {
    const auto& b_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    uint32_t batch_size_b = get_batch_size(b_shape_padded);
    bool broadcast_batch = batch_size_b == 1;
    if (!matmul_program_config.has_value()) {
        return broadcast_batch;
    }

    bool is_multi_core_reuse = std::visit(
        [](const auto& program_config) -> bool {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            return static_cast<bool>(
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig>);
        },
        matmul_program_config.value());
    if (is_multi_core_reuse) {
        const auto& a_shape_padded =
            operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
        uint32_t batch_size_a = get_batch_size(a_shape_padded);
        broadcast_batch &= batch_size_a > 1;
    }
    return broadcast_batch;
}

}  // namespace

MatmulDeviceOperation::program_factory_t MatmulDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    const auto& config = operation_attributes.program_config.value();

    return std::visit(
        [](const auto& c) -> program_factory_t {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreProgramConfig>) {
                return MatmulMultiCoreProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                return MatmulMultiCoreReuseOptimizedProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return MatmulMultiCoreReuseMcast2DProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                // gather_in0 uses the legacy MeshWorkload path (create_descriptor not yet supported)
                if (c.gather_in0) {
                    return MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory{};
                }
                return MatmulMultiCoreReuseMcast1DProgramFactory{};
            } else if constexpr (std::is_same_v<
                                     T,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory{};
            } else if constexpr (
                std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                return MatmulMultiCoreReuseBatchedHSDRAMShardedProgramFactory{};
            } else {
                TT_THROW("Unknown program config type");
            }
        },
        config);
}

// ===========================================================================
// Entry point. Runs the universal validators, then the config-based shared validators,
// then the per-config dispatch (std::visit) below.
// ===========================================================================
void MatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    using namespace tt::constants;
    warn_if_allowed_worker_cores_missing(
        attributes.program_config, "MatmulDeviceOperation::validate_on_program_cache_miss");

    const auto& input_tensors = args.input_tensors;
    const auto& input_tensor_a = args.input_tensors.at(0);
    const auto& input_tensor_b = args.input_tensors.at(1);
    const auto& optional_input_tensors = args.optional_input_tensors;

    const auto& a_shape =
        operations::matmul::utilities::get_matmul_tensor_logical_shape(input_tensor_a, attributes.transpose_a);
    const auto& b_shape =
        operations::matmul::utilities::get_matmul_tensor_logical_shape(input_tensor_b, attributes.transpose_b);
    const auto& a_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_a, attributes.transpose_a);
    const auto& b_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_b, attributes.transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_a, attributes.transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_b, attributes.transpose_b);

    // ---- universal checks, part 1: independent of the chosen program config ----
    validate_matmul_operand_basics(input_tensor_a, input_tensor_b, in0_tile, in1_tile);
    validate_matmul_matrix_dimensions(a_shape, b_shape, a_shape_padded, b_shape_padded, in0_tile, in1_tile);
    validate_matmul_bfloat4_tile_dims(input_tensor_a, input_tensor_b, in0_tile, in1_tile);
    validate_matmul_optional_tensors(attributes, args);

    // ---- choose + normalize the program config ----
    const auto& optional_bias = optional_input_tensors.at(0);
    uint32_t bias_single_tile_size = 0;
    if (optional_bias.has_value()) {
        auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(optional_bias.value().dtype());
        bias_single_tile_size = tt::tile_size(bias_data_format);
    }
    operations::matmul::MatmulProgramConfig chosen_program_config = operations::matmul::get_program_config(
        input_tensor_a,
        input_tensor_b,
        attributes.transpose_a,
        attributes.transpose_b,
        bias_single_tile_size,
        attributes);
    operations::matmul::normalize_program_config(
        chosen_program_config, input_tensor_a.device()->compute_with_storage_grid_size());

    // ---- universal checks, part 2: need the chosen program config ----
    // Config-based shared validators (each self-filters by config via std::visit).
    validate_matmul_tiny_tile_constraints(input_tensor_b, in0_tile, in1_tile, chosen_program_config);
    validate_matmul_compute_grid_and_per_core_dims(input_tensor_a, chosen_program_config);
    validate_matmul_block_and_subblock_configuration(attributes, a_shape_padded, in0_tile, chosen_program_config);
    validate_matmul_sharded_operand_grids_within_program_compute_grid(
        input_tensor_a, input_tensor_b, chosen_program_config);
    validate_matmul_reuse_sharded_output_block_divisibility(
        input_tensor_a, input_tensor_b, a_shape_padded, b_shape_padded, in0_tile, in1_tile, chosen_program_config);
    validate_matmul_work_distribution_and_gather_ring_topology(
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

    validate_matmul_batch_compatibility(
        attributes, input_tensor_a, input_tensor_b, a_shape, b_shape, chosen_program_config);
    validate_matmul_mcast1d_subdevice_worker_grid(input_tensor_a, attributes, chosen_program_config);
    validate_matmul_input_count(attributes, input_tensors, input_tensor_b, chosen_program_config);
    validate_matmul_bias_shape(
        optional_bias, in0_tile, in1_tile, a_shape_padded, b_shape, b_shape_padded, chosen_program_config);
    validate_matmul_untilize_out(attributes, chosen_program_config);

    // ---- per-config validation: one std::visit over the chosen program config ----
    // Shared Mcast2D/Mcast1D fuse_batch preamble first, then dispatch to the matching
    // per-config validator. The Reuse work-split check runs after the visit.
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                validate_matmul_mcast_fuse_batch_preamble(
                    ttsl::get_type_name(program_config),
                    program_config.fuse_batch,
                    attributes,
                    a_shape_padded,
                    b_shape_padded,
                    in0_tile);
            }
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                validate_matmul_mcast1d_config(
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
                validate_matmul_dram_sharded_config(
                    input_tensor_a, input_tensor_b, attributes, a_shape_padded, in0_tile, program_config);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::
                                         MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                validate_matmul_batched_dram_sharded_config(
                    input_tensor_a, input_tensor_b, attributes, a_shape_padded, in0_tile, program_config);
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                validate_matmul_mcast2d_config(
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
                validate_matmul_reuse_config(
                    input_tensor_a,
                    input_tensor_b,
                    attributes,
                    a_shape_padded,
                    b_shape_padded,
                    in0_tile,
                    in1_tile,
                    program_config);
            } else {
                validate_matmul_multicore_config(input_tensor_a, input_tensor_b, attributes, in0_tile, in1_tile);
            }
        },
        chosen_program_config);

    if (std::holds_alternative<operations::matmul::MatmulMultiCoreReuseProgramConfig>(chosen_program_config)) {
        operations::matmul::utilities::validate_matmul_reuse_work_split(
            input_tensor_a,
            input_tensor_b,
            a_shape_padded,
            b_shape_padded,
            in0_tile,
            in1_tile,
            std::get<operations::matmul::MatmulMultiCoreReuseProgramConfig>(chosen_program_config),
            attributes.output_mem_config,
            std::nullopt);
    }
}

MatmulDeviceOperation::spec_return_value_t MatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    using namespace tt::tt_metal;
    using namespace tt::constants;
    warn_if_allowed_worker_cores_missing(attributes.program_config, "MatmulDeviceOperation::compute_output_specs");
    const auto& optional_output_tensors = args.optional_output_tensors;
    const auto& input_tensors = args.input_tensors;
    const auto& optional_input_tensors = args.optional_input_tensors;

    TT_FATAL(
        optional_output_tensors.size() <= 1,
        "None or One Optional output tensor can be passed when accessing it "
        "for computing Matmul's output specs");

    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();

    if (is_optional_output_tensor) {
        return {optional_output_tensors.at(0)->tensor_spec()};
    }

    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    // Use the compute_matmul_output_shape function to get the output shape
    const auto output_shape = operations::matmul::utilities::compute_matmul_output_shape(
        input_tensor_a, input_tensor_b, attributes.transpose_a, attributes.transpose_b);

    const auto& a_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_a, attributes.transpose_a);
    const auto& b_shape_padded =
        operations::matmul::utilities::get_matmul_tensor_padded_shape(input_tensor_b, attributes.transpose_b);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_a, attributes.transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_b, attributes.transpose_b);
    auto output_tile = attributes.output_tile.value();
    auto tile_width_ratio = output_tile.get_tile_shape()[1] / in1_tile.get_width();
    auto output_layout = attributes.untilize_out ? Layout::ROW_MAJOR : Layout::TILE;

    TT_FATAL(
        attributes.output_dtype.has_value(), "output_dtype must be populated before computing matmul output specs");
    if (attributes.output_mem_config.is_sharded()) {
        const auto& optional_bias = !optional_input_tensors.empty() && optional_input_tensors[0].has_value()
                                        ? optional_input_tensors[0]
                                        : std::nullopt;
        uint32_t bias_single_tile_size = 0;
        if (optional_bias.has_value()) {
            auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(optional_bias.value().dtype());
            bias_single_tile_size = tt::tile_size(bias_data_format);
        }
        operations::matmul::MatmulProgramConfig chosen_program_config = operations::matmul::get_program_config(
            input_tensor_a,
            input_tensor_b,
            attributes.transpose_a,
            attributes.transpose_b,
            bias_single_tile_size,
            attributes);
        // Soft-normalize so downstream variant code can safely read allowed_worker_cores; the warning
        // for missing allowed_worker_cores was emitted at the entry point above.
        operations::matmul::normalize_program_config(
            chosen_program_config, input_tensor_a.device()->compute_with_storage_grid_size());
        return std::visit(
            [&](const auto& program_config) -> MatmulDeviceOperation::spec_return_value_t {
                using ProgramConfigType = std::decay_t<decltype(program_config)>;
                if constexpr (std::is_same_v<
                                  ProgramConfigType,
                                  operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                    const auto M =
                        operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, program_config.fuse_batch);
                    const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N ({}) must be a multiple of the output/in1 tile-width ratio ({}) so "
                        "columns fill whole output tiles",
                        per_core_N,
                        tile_width_ratio);
                    auto mem_config = attributes.output_mem_config;
                    if (!program_config.gather_in0) {
                        // Check if BLOCK_SHARDED on 1D grid - if so, use user's shard spec with converted memory layout
                        auto memory_layout = mem_config.memory_layout();
                        bool is_block_sharded_1d = false;
                        if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED && mem_config.shard_spec().has_value()) {
                            auto grid_bbox = mem_config.shard_spec()->grid.bounding_box();
                            bool is_1d_column = (grid_bbox.end_coord.x == grid_bbox.start_coord.x);
                            bool is_1d_row = (grid_bbox.end_coord.y == grid_bbox.start_coord.y);
                            is_block_sharded_1d = is_1d_column || is_1d_row;
                        }

                        if (is_block_sharded_1d) {
                            // Use user's shard spec with converted memory layout
                            auto user_shard_spec = mem_config.shard_spec().value();
                            auto grid_bbox = user_shard_spec.grid.bounding_box();
                            bool is_1d_column = (grid_bbox.end_coord.x == grid_bbox.start_coord.x);
                            memory_layout =
                                is_1d_column ? TensorMemoryLayout::HEIGHT_SHARDED : TensorMemoryLayout::WIDTH_SHARDED;
                            mem_config =
                                tt::tt_metal::MemoryConfig{memory_layout, mem_config.buffer_type(), user_shard_spec};
                        } else {
                            // Compute shard spec from per_core values
                            uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                            uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                            uint32_t num_cores = num_blocks_x * num_blocks_y;
                            auto cwsg_1d = program_config.allowed_worker_cores.value().bounding_box().grid_size();
                            CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, cwsg_1d, true);
                            tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec{
                                all_cores,
                                {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                                ShardOrientation::ROW_MAJOR};
                            mem_config = tt::tt_metal::MemoryConfig(
                                mem_config.memory_layout(), mem_config.buffer_type(), shard_spec);
                        }
                    }
                    // support for multi-tensor output
                    const ttnn::TensorSpec tensor_spec(
                        output_shape,
                        tt::tt_metal::TensorLayout(
                            attributes.output_dtype.value(),
                            attributes.untilize_out ? tt::tt_metal::PageConfig(output_layout)
                                                    : tt::tt_metal::PageConfig(output_layout, output_tile),
                            mem_config));

                    std::vector<ttnn::TensorSpec> output_tensor_specs(input_tensors.size() - 1, tensor_spec);
                    return output_tensor_specs;
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                    const auto M =
                        operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
                    const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
                    const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);

                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;
                    uint32_t per_core_K = input_tensor_a.shard_spec().value().shape[1] / in0_tile.get_width();

                    TT_FATAL(
                        K % per_core_K == 0,
                        "in DRAM sharded Matmul we don't have support for un-even sharding currently. K: {}, "
                        "per_core_K: {}.",
                        K,
                        per_core_K);

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N ({}) must be a multiple of the output/in1 tile-width ratio ({}) so "
                        "columns fill whole output tiles",
                        per_core_N,
                        tile_width_ratio);

                    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    auto grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
                    CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid_size, true);
                    ShardSpec shard_spec = ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                        ShardOrientation::ROW_MAJOR};
                    auto mem_config = tt::tt_metal::MemoryConfig(
                        attributes.output_mem_config.memory_layout(),
                        attributes.output_mem_config.buffer_type(),
                        shard_spec);
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(
                            attributes.output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::
                                             MatmulMultiCoreReuseMultiCastBatchedDRAMShardedProgramConfig>) {
                    // For batched DRAM sharded matmul, use the user-provided output shard spec
                    TT_FATAL(
                        attributes.output_mem_config.shard_spec().has_value(),
                        "Output memory config must have a shard spec for batched DRAM sharded matmul");

                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N ({}) must be a multiple of the output/in1 tile-width ratio ({}) so "
                        "columns fill whole output tiles",
                        per_core_N,
                        tile_width_ratio);

                    // Use the user-provided shard spec directly
                    auto mem_config = attributes.output_mem_config;
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(
                            attributes.output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                    const auto M =
                        operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
                    const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N ({}) must be a multiple of the output/in1 tile-width ratio ({}) so "
                        "columns fill whole output tiles",
                        per_core_N,
                        tile_width_ratio);

                    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                    // The output CB is globally allocated against the output tensor on the factory's
                    // work grid {start_core, start_core + num_blocks - 1}, so the output shard grid
                    // computed here must match it exactly. Mirror the factory's start_core derivation
                    // (allowed_worker_cores is the single source of truth for core placement) rather
                    // than trusting a user-supplied output shard grid, which need not agree.
                    const CoreCoord start_core =
                        program_config.allowed_worker_cores.has_value()
                            ? program_config.allowed_worker_cores.value().bounding_box().start_coord
                            : CoreCoord{0, 0};
                    CoreRangeSet all_cores;
                    ShardOrientation shard_orientation;
                    if (program_config.transpose_mcast) {
                        all_cores = CoreRangeSet({CoreRange(
                            start_core, {start_core.x + num_blocks_y - 1, start_core.y + num_blocks_x - 1})});
                        shard_orientation = ShardOrientation::COL_MAJOR;
                    } else {
                        all_cores = CoreRangeSet({CoreRange(
                            start_core, {start_core.x + num_blocks_x - 1, start_core.y + num_blocks_y - 1})});
                        shard_orientation = ShardOrientation::ROW_MAJOR;
                    }
                    tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                        shard_orientation};
                    auto mem_config = tt::tt_metal::MemoryConfig(
                        attributes.output_mem_config.memory_layout(),
                        attributes.output_mem_config.buffer_type(),
                        shard_spec);
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(
                            attributes.output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                    const auto M =
                        operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
                    const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                    uint32_t per_core_M = program_config.per_core_M;
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        per_core_N % tile_width_ratio == 0,
                        "per_core_N ({}) must be a multiple of the output/in1 tile-width ratio ({}) so "
                        "columns fill whole output tiles",
                        per_core_N,
                        tile_width_ratio);

                    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
                    if (input_tensor_a.is_sharded()) {
                        shard_orientation = input_tensor_a.shard_spec().value().orientation;
                    } else if (input_tensor_b.is_sharded()) {
                        shard_orientation = input_tensor_b.shard_spec().value().orientation;
                    }

                    CoreRangeSet all_cores;
                    if (attributes.output_mem_config.shard_spec().has_value()) {
                        all_cores = attributes.output_mem_config.shard_spec()->grid;
                    } else {
                        auto cwsg_2d = program_config.allowed_worker_cores.value().bounding_box().grid_size();
                        all_cores = num_cores_to_corerangeset(
                            num_cores, cwsg_2d, shard_orientation == ShardOrientation::ROW_MAJOR);
                    }
                    tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                        shard_orientation};
                    auto mem_config = tt::tt_metal::MemoryConfig(
                        attributes.output_mem_config.memory_layout(),
                        attributes.output_mem_config.buffer_type(),
                        shard_spec);
                    return {TensorSpec(
                        output_shape,
                        TensorLayout(
                            attributes.output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else {
                    TT_FATAL(
                        in0_tile.get_height() == TILE_HEIGHT and in0_tile.get_width() == TILE_WIDTH,
                        "matmul with non-optimized program config does not "
                        "support tiny tile");
                    TT_FATAL(
                        in1_tile.get_height() == TILE_HEIGHT and in1_tile.get_width() == TILE_WIDTH,
                        "matmul with non-optimized program config does not "
                        "support tiny tile");
                    if (attributes.output_tile.has_value()) {
                        TT_FATAL(
                            attributes.output_tile->get_tile_shape()[0] == TILE_HEIGHT and
                                attributes.output_tile->get_tile_shape()[1] == TILE_WIDTH,
                            "matmul with non-optimized program config does not "
                            "support tiny tile");
                    }
                    TT_THROW("Unsupported op for output sharding");
                    ttsl::unreachable();
                }
            },
            chosen_program_config);
    }

    return {TensorSpec(
        output_shape,
        TensorLayout(
            attributes.output_dtype.value(),
            PageConfig(Layout::TILE, attributes.output_tile),
            attributes.output_mem_config))};
}

MatmulDeviceOperation::tensor_return_value_t MatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    warn_if_allowed_worker_cores_missing(attributes.program_config, "MatmulDeviceOperation::create_output_tensors");
    const auto& optional_output_tensors = args.optional_output_tensors;
    const auto& input_tensors = args.input_tensors;
    tensor_return_value_t output_tensors;

    if (!optional_output_tensors.empty() and optional_output_tensors[0].has_value()) {
        output_tensors.reserve(optional_output_tensors.size());
        for (const auto& optional_output_tensor : optional_output_tensors) {
            TT_FATAL(
                optional_output_tensor.has_value(),
                "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(optional_output_tensor.value());
        }
        return output_tensors;
    }
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = compute_output_specs(attributes, args);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
    return output_tensors;
}

ttsl::hash::hash_t MatmulDeviceOperation::compute_descriptor_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    const auto& input_tensors = args.input_tensors;
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    auto factory = select_program_factory(attributes, args);

    auto hash = tt::tt_metal::operation::hash_operation<MatmulDeviceOperation>(
        attributes, factory.index(), input_tensor_a, input_tensor_b);

    for (const auto& optional_input_tensor : args.optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            hash = ttsl::hash::hash_objects(hash, optional_input_tensor.value());
        }
    }

    for (const auto& optional_output_tensor : args.optional_output_tensors) {
        if (optional_output_tensor.has_value()) {
            hash = ttsl::hash::hash_objects(hash, optional_output_tensor.value());
        }
    }
    return hash;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<MatmulDeviceOperation::tensor_return_value_t>
MatmulDeviceOperation::create_op_performance_model(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    using namespace tt::tt_metal;
    const auto& input_tensor_a = tensor_args.input_tensors.at(0);
    const auto& input_tensor_b = tensor_args.input_tensors.at(1);

    const auto& in_a_shape = input_tensor_a.logical_shape();
    const auto& out_shape = output_tensors.at(0).logical_shape();

    const auto& t = output_tensors.at(0);
    if (t.storage_type() != StorageType::DEVICE) {
        log_warning(tt::LogOp, "Output tensor not on DEVICE?!");
    }

    const CoreCoord compute_grid = t.device()->compute_with_storage_grid_size();
    const int num_cores = compute_grid.x * compute_grid.y;
    // The Wormhole/Blackhole matrix engine performs 8x16 x 16x16 = 8x16 in a single cycle.
    // This is 2*8*16*16 = 4096 muladds in a single cycle.
    constexpr int tensix_mul_adds_per_cycle_lofi = 4096;

    // Calculate number of mul/add operations
    // TODO: add bias modeling
    int64_t num_mul_adds_per_elem = in_a_shape[-1] * 2;  // 1 multiply and 1 add per element
    uint32_t batch_size = get_batch_size(out_shape);
    int64_t num_mul_adds = num_mul_adds_per_elem * out_shape[-2] * out_shape[-1] * batch_size;

    MathFidelity math_fidelity = ttnn::get_math_fidelity(operation_attributes.compute_kernel_config);

    int ideal_dev_clock_cycles = std::ceil(
        ((float)num_mul_adds / (float)(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        (float)operation::OpPerformanceModel::fidelity_multiplier(math_fidelity));

    operation::OpPerformanceModelGeneral<MatmulDeviceOperation::tensor_return_value_t> result(
        {input_tensor_a, input_tensor_b}, output_tensors, ideal_dev_clock_cycles);

    return result;
}

MatmulParams create_matmul_attributes(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const MatmulParams& parameters,
    const std::vector<std::optional<Tensor>>& optional_output_tensors) {
    tt::tt_metal::IDevice* device = input_tensor_a.device();
    TT_FATAL(device != nullptr, "Operand to matmul must be on device");
    auto arch = device->arch();
    const bool has_user_grid = parameters.user_core_coord.has_value();
    const bool has_program_config = parameters.program_config.has_value();
    bool are_inputs_low_precision_df =
        ((input_tensor_a.dtype() == DataType::BFLOAT8_B || input_tensor_a.dtype() == DataType::BFLOAT4_B) &&
         (input_tensor_b.dtype() == DataType::BFLOAT8_B || input_tensor_b.dtype() == DataType::BFLOAT4_B));
    const auto increase_fidelity = !has_program_config && !has_user_grid && !are_inputs_low_precision_df;
    auto math_fidelity = increase_fidelity ? MathFidelity::HiFi2 : MathFidelity::LoFi;
    bool are_inputs_32F = (input_tensor_a.dtype() == DataType::FLOAT32 && input_tensor_b.dtype() == DataType::FLOAT32);
    // Due to hardware bug (#38306), HiFi4 + fp32_dest_acc_en can sometime produce incorrect results on Wormhole.
    // When inputs are FLOAT32 (which drives fp32_dest_acc_en=True by default), use HiFi3 on Wormhole B0.
    const auto is_wormhole = arch == tt::ARCH::WORMHOLE_B0;
    math_fidelity = are_inputs_32F ? (is_wormhole ? MathFidelity::HiFi3 : MathFidelity::HiFi4) : math_fidelity;

    bool broadcast_batch = parameters.bcast_batch.value_or(get_broadcast_batch(
        input_tensor_a, input_tensor_b, parameters.transpose_a, parameters.transpose_b, parameters.program_config));
    TT_FATAL(!(has_user_grid && has_program_config), "Cannot use both user core grid/coordinates and a program config");

    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();
    std::optional<DataType> output_dtype = parameters.output_dtype;
    MemoryConfig output_mem_config = parameters.output_mem_config;

    if (is_optional_output_tensor) {
        const auto& optional_output_tensor = optional_output_tensors.at(0);
        if (output_mem_config == tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG) {
            output_mem_config = optional_output_tensor->memory_config();
        } else {
            TT_FATAL(
                optional_output_tensor->memory_config() == output_mem_config,
                "Memory config mismatch between optional output tensor {} & "
                "output tensor {}",
                optional_output_tensor->memory_config(),
                output_mem_config);
        }

        if (output_dtype.has_value()) {
            TT_FATAL(
                optional_output_tensor->dtype() == output_dtype.value(),
                "Type mismatch between optional output tensor {} & output tensor {}",
                optional_output_tensor->dtype(),
                output_dtype.value());
        } else {
            output_dtype = optional_output_tensor->dtype();
        }
    } else {
        if (!output_dtype.has_value()) {
            output_dtype = input_tensor_a.dtype();
        }
    }
    bool is_float_32 = output_dtype == DataType::FLOAT32;
    auto kernel_config_val = init_device_compute_kernel_config(
        arch,
        parameters.compute_kernel_config,
        math_fidelity,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/is_float_32,
        /*default_l1_acc=*/!is_float_32);
    ttnn::verify_numerical_configuration(arch, parameters.compute_kernel_config);
    auto in0_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_a, parameters.transpose_a);
    auto in1_tile = operations::matmul::utilities::get_matmul_tile(input_tensor_b, parameters.transpose_b);

    std::optional<tt::tt_metal::Tile> optional_output_tensor_tile = std::nullopt;
    if (is_optional_output_tensor) {
        optional_output_tensor_tile = optional_output_tensors.at(0)->tensor_spec().tile();
    }
    tt::tt_metal::Tile output_tile = operations::matmul::utilities::get_output_tile(
        output_mem_config, in0_tile, in1_tile, parameters.output_tile, optional_output_tensor_tile);

    return MatmulParams{
        parameters.program_config,
        broadcast_batch,
        output_mem_config,
        output_dtype,
        kernel_config_val,
        parameters.untilize_out,
        parameters.user_core_coord,
        parameters.user_fused_activation,
        parameters.user_run_batched,
        parameters.transpose_a,
        parameters.transpose_b,
        output_tile,
        parameters.global_cb,
        parameters.sub_device_id};
}

MatmulDeviceOperation::tensor_return_value_t matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& optional_output_tensor,
    const MatmulParams& attributes) {
    MatmulParams normalized_attributes = attributes;
    if (!normalized_attributes.program_config.has_value()) {
        uint32_t bias_single_tile_size = 0;
        if (bias.has_value()) {
            auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(bias.value().dtype());
            bias_single_tile_size = tt::tile_size(bias_data_format);
        }

        normalized_attributes.program_config = operations::matmul::get_program_config(
            input_tensor_a,
            input_tensor_b,
            normalized_attributes.transpose_a,
            normalized_attributes.transpose_b,
            bias_single_tile_size,
            normalized_attributes);
    }
    operations::matmul::normalize_program_config(
        normalized_attributes.program_config.value(), input_tensor_a.device()->compute_with_storage_grid_size());
    return ttnn::device_operation::launch<MatmulDeviceOperation>(
        normalized_attributes, {{input_tensor_a, input_tensor_b}, {bias}, {optional_output_tensor}});
}

MatmulDeviceOperation::tensor_return_value_t matmul(
    const std::vector<Tensor>& input_tensors,
    const std::optional<Tensor>& optional_output_tensor,
    const MatmulParams& attributes) {
    MatmulParams normalized_attributes = attributes;
    if (!normalized_attributes.program_config.has_value()) {
        uint32_t bias_single_tile_size = 0;

        normalized_attributes.program_config = operations::matmul::get_program_config(
            input_tensors.at(0),
            input_tensors.at(1),
            normalized_attributes.transpose_a,
            normalized_attributes.transpose_b,
            bias_single_tile_size,
            normalized_attributes);
    }
    operations::matmul::normalize_program_config(
        normalized_attributes.program_config.value(), input_tensors.at(0).device()->compute_with_storage_grid_size());
    return ttnn::device_operation::launch<MatmulDeviceOperation>(
        normalized_attributes, {input_tensors, {}, {optional_output_tensor}});
}

}  // namespace ttnn::prim

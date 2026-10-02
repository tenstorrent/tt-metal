// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"

#include <string_view>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/matmul_validation.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "tt-metalium/hal_types.hpp"
#include "tt-metalium/experimental/global_circular_buffer.hpp"
#include "ttnn/global_circular_buffer.hpp"
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
        [&operation_attributes](const auto& c) -> program_factory_t {
            using T = std::decay_t<decltype(c)>;
            if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreProgramConfig>) {
                return MatmulMultiCoreProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                return MatmulMultiCoreReuseOptimizedProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return MatmulMultiCoreReuseMcast2DProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                // gather_in0 (create_descriptor not yet supported) and any GCB-backed config
                // (ProgramDescriptor cannot attach an experimental GlobalCircularBuffer) use the legacy
                // MeshWorkload builder.
                if (c.gather_in0 || operation_attributes.global_cb.has_value()) {
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
// Entry point. Runs the universal validators, chooses the program config, then throws the first
// rule the config breaks (program_config_error, matmul_validation.hpp).
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
    operations::matmul::MatmulProgramConfig chosen_program_config = operations::matmul::get_program_config(
        input_tensor_a, input_tensor_b, attributes.transpose_a, attributes.transpose_b, optional_bias, attributes);
    operations::matmul::normalize_program_config(
        chosen_program_config, input_tensor_a.device()->compute_with_storage_grid_size());

    // ---- checks that need the chosen program config ----
    if (auto error =
            program_config_error(matmul_specs(input_tensors, optional_bias, attributes), chosen_program_config);
        !error.empty()) {
        TT_THROW("{}", error);
    }
    // The weight ↔ matmul cross-checks of mcast_in0 with a global CB (per-receiver shard geometry,
    // K % in0_block_w == 0, per_core_N == per-receiver N, stream_in1 == false) are owned by the shared prefetcher
    // helper, which dispatches on the weight's detected DRAM layout — receiver-contiguous NdShardSpec or legacy
    // K-row-major WIDTH_SHARDED — and reads the weight's buffer.
    if (const auto* config_1d =
            std::get_if<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(&chosen_program_config);
        config_1d != nullptr && attributes.global_cb.has_value() && !config_1d->gather_in0 && config_1d->mcast_in0) {
        ttnn::global_circular_buffer::tensor_prefetcher_block_count_for_matmul_1d(
            *config_1d, input_tensor_b, attributes.global_cb.value());
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
        operations::matmul::MatmulProgramConfig chosen_program_config = operations::matmul::get_program_config(
            input_tensor_a, input_tensor_b, attributes.transpose_a, attributes.transpose_b, optional_bias, attributes);
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
                    const tt::tt_metal::TensorSpec tensor_spec(
                        output_shape,
                        tt::tt_metal::TensorLayout(
                            attributes.output_dtype.value(),
                            attributes.untilize_out ? tt::tt_metal::PageConfig(output_layout)
                                                    : tt::tt_metal::PageConfig(output_layout, output_tile),
                            mem_config));

                    std::vector<tt::tt_metal::TensorSpec> output_tensor_specs(input_tensors.size() - 1, tensor_spec);
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
                    return {tt::tt_metal::TensorSpec(
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
                    return {tt::tt_metal::TensorSpec(
                        output_shape,
                        TensorLayout(
                            attributes.output_dtype.value(), PageConfig(output_layout, output_tile), mem_config))};
                } else if constexpr (std::is_same_v<
                                         ProgramConfigType,
                                         operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
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

                    const uint32_t B = program_config.fuse_batch ? 1u : get_batch_size(a_shape_padded);
                    operations::matmul::utilities::validate_block_sharded_output_batch(true, B, per_core_M, per_core_N);

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
                    return {tt::tt_metal::TensorSpec(
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
                    return {tt::tt_metal::TensorSpec(
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

    return {tt::tt_metal::TensorSpec(
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
    tt::tt_metal::distributed::MeshDevice* device = input_tensor_a.device();
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
        normalized_attributes.program_config = operations::matmul::get_program_config(
            input_tensor_a,
            input_tensor_b,
            normalized_attributes.transpose_a,
            normalized_attributes.transpose_b,
            bias,
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
        normalized_attributes.program_config = operations::matmul::get_program_config(
            input_tensors.at(0),
            input_tensors.at(1),
            normalized_attributes.transpose_a,
            normalized_attributes.transpose_b,
            std::nullopt,
            normalized_attributes);
    }
    operations::matmul::normalize_program_config(
        normalized_attributes.program_config.value(), input_tensors.at(0).device()->compute_with_storage_grid_size());
    // validate requires optional_input_tensors.size() == 1; this path has no bias.
    return ttnn::device_operation::launch<MatmulDeviceOperation>(
        normalized_attributes, {input_tensors, {std::nullopt}, {optional_output_tensor}});
}

}  // namespace ttnn::prim

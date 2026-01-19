// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "tt-metalium/work_split.hpp"
#include "tt_stl/unreachable.hpp"

namespace ttnn::prim {

namespace {

void check_tensor_in_grid(const Tensor& tensor, const CoreCoord& grid_size) {
    // Validate tensor is within grid if sharded and not in DRAM
    if (tensor.memory_config().is_sharded() && tensor.memory_config().buffer_type() != BufferType::DRAM) {
        const auto& shard_spec = tensor.memory_config().shard_spec().value();
        const auto& shard_grid = shard_spec.grid;
        CoreRange range(CoreCoord(0, 0), grid_size);
        TT_FATAL(
            range.contains(shard_grid),
            "Tensor shard spec grid must be within config grid! Shard grid: {}, Config grid: {}",
            shard_grid,
            range);
    }
}

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
                return MatmulMeshWorkloadMultiCoreFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                return MatmulMeshWorkloadMultiCoreReuseOptimizedProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                return MatmulMeshWorkloadMultiCoreReuseMcast2DProgramFactory{};
            } else if constexpr (std::is_same_v<T, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                return MatmulMeshWorkloadMultiCoreReuseMcast1DProgramFactory{};
            } else if constexpr (std::is_same_v<
                                     T,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                return MatmulMultiCoreReuseMultiCastDRAMShardedProgramFactory{};
            } else {
                TT_THROW("Unknown program config type");
            }
        },
        config);
}

void MatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    using namespace tt::constants;

    const auto& input_tensors = args.input_tensors;
    const auto& input_tensor_a = args.input_tensors.at(0);
    const auto& input_tensor_b = args.input_tensors.at(1);
    const auto& optional_output_tensors = args.optional_output_tensors;
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
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to matmul need to be on device!");
    TT_FATAL(
        (in0_tile.get_width() == TILE_WIDTH && in1_tile.get_height() == TILE_WIDTH),
        "Input tile dims must have inner dim equal to 32 due to llk constraints");

    TT_FATAL(
        (input_tensor_a.layout() == Layout::TILE && input_tensor_b.layout() == Layout::TILE),
        "Inputs to matmul must be tilized");
    TT_FATAL(
        a_shape[-1] == b_shape[-2],
        "The width of the first tensor must be equal to the height of the "
        "second tensor. Mismatch: width={} height={}",
        a_shape[-1],
        b_shape[-2]);

    const bool is_optional_output_tensor =
        !optional_output_tensors.empty() && optional_output_tensors.at(0).has_value();

    TT_FATAL(
        optional_input_tensors.size() == 1,
        "Must have exactly 1 optional input tensor, got: {}",
        optional_input_tensors.size());

    const auto output_tensor_spec = compute_output_specs(attributes, args).at(0);
    if (is_optional_output_tensor) {
        const auto& optional_output_tensor_c = optional_output_tensors.at(0);
        const auto& optional_output_tensor_shape = optional_output_tensor_c->logical_shape();
        TT_FATAL(
            optional_output_tensor_shape == output_tensor_spec.logical_shape(),
            "Shape of Optional Output Tensor {} doesnt match Output Tensor {}",
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
        TT_FATAL(
            output_tensor_spec.memory_config().memory_layout() == attributes.output_mem_config.memory_layout(),
            "Mismatch between computed {} and provided {} mem config memory layout",
            output_tensor_spec.memory_config().memory_layout(),
            attributes.output_mem_config.memory_layout());
        TT_FATAL(
            output_tensor_spec.memory_config().buffer_type() == attributes.output_mem_config.buffer_type(),
            "Mismatch between computed {} and provided {} mem config buffer type",
            output_tensor_spec.memory_config().buffer_type(),
            attributes.output_mem_config.buffer_type());
        if (attributes.output_mem_config.shard_spec().has_value() &&
            output_tensor_spec.memory_config() != attributes.output_mem_config) {
            log_warning(
                tt::LogOp,
                "Mismatch between computed {} and provided {} mem config. Using computed config.",
                output_tensor_spec.memory_config(),
                attributes.output_mem_config);
        }
    }

    TT_FATAL(attributes.bcast_batch.has_value(), "Error: bcast_batch field should have been automatically populated");
    TT_FATAL(attributes.output_tile.has_value(), "Error: output_tile field should have been automatically populated");
    if (attributes.bcast_batch.value()) {
        TT_FATAL(
            get_batch_size(b_shape) == 1,
            "The batch bcast variant of matmul requires input tensors of shapes BCMK*11KN=BCMN "
            "or equivalent. Please change the second input tensor or adjust the program config.");
    } else {
        // same condition as above, different message
        TT_FATAL(
            a_shape.rank() == b_shape.rank() && "bmm (non-bcast matmul) expects input tensors of the same rank",
            "Input tensors must have the same rank, got a_shape rank: {} vs b_shape rank: {}",
            a_shape.rank(),
            b_shape.rank());
        for (auto i = 0; i < a_shape.rank() - 2; i++) {
            TT_FATAL(
                a_shape[i] == b_shape[i],
                "bmm (non-bcast matmul) expects input tensors of shapes "
                "BCMK*BCKN=BCMN or equivalent");
        }
    }

    TT_FATAL(is_floating_point(input_tensor_a.dtype()), "Unsupported data format");

    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to matmul need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to matmul need to be on the same device!");

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

    if (std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
            chosen_program_config) &&
        attributes.global_cb.has_value() && input_tensor_b.is_sharded() && input_tensor_b.buffer()->is_dram()) {
        for (uint32_t i = 1; i < input_tensors.size(); ++i) {
            TT_FATAL(
                input_tensor_b.logical_shape() == input_tensors[i].logical_shape(),
                "for multi-tensor matmul, all weight tensors must have the same logical_shape, {} is not equal to {}",
                input_tensor_b.logical_shape(),
                input_tensors[i].logical_shape());
            TT_FATAL(
                input_tensor_b.padded_shape() == input_tensors[i].padded_shape(),
                "for multi-tensor matmul, all weight tensors must have the same padded_shape {} is not equal to {}",
                input_tensor_b.padded_shape(),
                input_tensors[i].padded_shape());
            TT_FATAL(
                input_tensor_b.tensor_spec() == input_tensors[i].tensor_spec(),
                "for multi-tensor matmul, all weight tensors must have the same tensor_spec {} is not equal to {}",
                input_tensor_b.tensor_spec(),
                input_tensors[i].tensor_spec());
            TT_FATAL(
                input_tensor_b.layout() == input_tensors[i].layout(),
                "for multi-tensor matmul, all weight tensors must have the same layout {} is not equal to {}",
                input_tensor_b.layout(),
                input_tensors[i].layout());
            TT_FATAL(
                input_tensor_b.dtype() == input_tensors[i].dtype(),
                "for multi-tensor matmul, all weight tensors must have the same _dtype {} is not equal to {}",
                input_tensor_b.dtype(),
                input_tensors[i].dtype());
        }
    } else {
        TT_FATAL(input_tensors.size() == 2, "Must have exactly 2 input tensors, got: {}", input_tensors.size());
    }

    if (optional_bias.has_value()) {
        const auto& bias = optional_bias.value();
        auto bias_tile_shape = bias.tensor_spec().tile().get_tile_shape();
        TT_FATAL(
            (bias_tile_shape[0] == in0_tile.get_height() && bias_tile_shape[1] == in1_tile.get_width()),
            "Input tile dims must have inner dim equal to 32 due to llk "
            "constraints");
        TT_FATAL(bias.layout() == Layout::TILE, "Unsupported input layout");
        const auto& bias_shape = bias.logical_shape();
        const auto& bias_shape_padded = bias.padded_shape();
        uint32_t bias_batch_size = get_batch_size(bias_shape);
        TT_FATAL(bias_batch_size == 1, "Unsupported bias shape: batch size not equal to 1.");
        TT_FATAL(
            bias_shape_padded[-2] == in0_tile.get_height(),
            "Unsupported bias shape: padded second last dimension of bias, "
            "{}, not equal to tile height, {}",
            bias_shape_padded[-2],
            in0_tile.get_height());
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
    }

    if (attributes.untilize_out) {
        TT_FATAL(attributes.output_dtype.has_value(), "Output dtype must be specified when untilize_out is true");
        TT_FATAL(
            (attributes.output_dtype.value() == DataType::BFLOAT16) ||
                (attributes.output_dtype.value() == DataType::FLOAT32),
            "Unsupported data type: {}",
            attributes.output_dtype.value());
        TT_FATAL(
            std::holds_alternative<operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>(
                chosen_program_config),
            "Untilize out is not supported for this program config: {}",
            typeid(chosen_program_config).name());
    }

    using namespace tt;
    std::visit(
        [input_tensor_a,
         input_tensor_b,
         optional_bias,
         a_shape_padded,
         b_shape_padded,
         in0_tile,
         in1_tile,
         &attributes](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(program_config.in0_block_w != 0, "in0_block_w is 0, which is not valid");
                TT_FATAL(program_config.out_subblock_h != 0, "out_subblock_h is 0, which is not valid");
                TT_FATAL(program_config.out_subblock_w != 0, "out_subblock_w is 0, which is not valid");
                TT_FATAL(program_config.out_block_h != 0, "out_block_h is 0, which is not valid");
                TT_FATAL(program_config.out_block_w != 0, "out_block_w is 0, which is not valid");
                TT_FATAL(program_config.per_core_M != 0, "per_core_M is 0, which is not valid");
                TT_FATAL(program_config.per_core_N != 0, "per_core_N is 0, which is not valid");
                if (program_config.fuse_batch) {
                    TT_FATAL(
                        get_batch_size(b_shape_padded) == 1,
                        "Matmul with fused batch requires input tensors of shapes BCMK*11KN=BCMN "
                        "or equivalent. Please change the second input tensor or adjust the program config.");
                }
            }
            // TODO: For 1D and 2D mcasts, we don't check if tensor is single core
            // or single row/col We can uplift these variants to skip mcasting to
            // support single core (1D) or single row/col (2D)
            if constexpr (std::is_same_v<
                              ProgramConfigType,
                              operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    program_config.per_core_M % program_config.out_block_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_M,
                    program_config.out_block_h);
                TT_FATAL(
                    program_config.per_core_N % program_config.out_block_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_N,
                    program_config.out_block_w);
                TT_FATAL(
                    program_config.out_block_h % program_config.out_subblock_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_h,
                    program_config.out_subblock_h);
                TT_FATAL(
                    program_config.out_block_w % program_config.out_subblock_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_w,
                    program_config.out_subblock_w);
                TT_FATAL(
                    !(program_config.mcast_in0 && program_config.gather_in0),
                    "Matmul1D does not support mcast_in0 and gather_in0 at the "
                    "same time.");

                // Gather in0 specific validation
                if (program_config.gather_in0) {
                    TT_FATAL(
                        program_config.num_global_cb_receivers > 0, "Num global CB receivers must be greater than 0.");
                    TT_FATAL(
                        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                        "Input tensor A must be width sharded when using gather_in0.");
                    TT_FATAL(
                        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                            (input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                             input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::DRAM),
                        "Input tensor B must be width sharded or DRAM interleaved when using gather_in0.");
                    if (!attributes.global_cb.has_value() && input_tensor_b.is_sharded()) {
                        if (input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::L1) {
                            TT_FATAL(
                                input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                                "Input tensor A and B must be sharded on the same cores "
                                "when using gather_in0.");
                        }
                    }
                    TT_FATAL(
                        attributes.output_mem_config.is_sharded(),
                        "Output tensor must be sharded when using gather_in0.");
                    TT_FATAL(
                        attributes.output_mem_config.shard_spec().has_value(),
                        "Output shard spec must be provided when using gather_in0.");

                    if (!input_tensor_b.is_sharded()) {
                        TT_FATAL(
                            !attributes.global_cb.has_value(),
                            "Global CB is not supported for DRAM_INTERLEAVED in1 when using gather_in0.");
                        TT_FATAL(
                            input_tensor_b.layout() == Layout::TILE,
                            "Input tensor B must be TILE_LAYOUT when DRAM_INTERLEAVED when using gather_in0.");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().grid ==
                                attributes.output_mem_config.shard_spec().value().grid,
                            "Input tensor A and output tensor must be sharded on the same cores when using gather_in0 "
                            "and in1 is DRAM_INTERLEAVED.");
                    }

                    if (!attributes.global_cb.has_value()) {
                        TT_FATAL(
                            program_config.num_global_cb_receivers == 1,
                            "Num global CB receivers must be 1 when global CB is not provided.");
                    }

                    TT_FATAL(!optional_bias.has_value(), "Bias is not supported when using gather_in0.");
                } else {
                    // Checks specific to non-gather configs
                    check_tensor_in_grid(input_tensor_a, program_config.compute_with_storage_grid_size);
                    check_tensor_in_grid(input_tensor_b, program_config.compute_with_storage_grid_size);
                }
                if (program_config.mcast_in0 || program_config.gather_in0) {
                    if (input_tensor_a.is_sharded()) {
                        TT_FATAL(program_config.fuse_batch, "Error: Batch fusion must be enabled.");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                            "Error: input_tensor_a must be width sharded. Provided tensor memory layout: {}",
                            input_tensor_a.memory_config().memory_layout());
                        if (attributes.output_mem_config.is_sharded()) {
                            TT_FATAL(
                                input_tensor_a.memory_config().buffer_type() ==
                                    attributes.output_mem_config.buffer_type(),
                                "Error: Buffer type mismatch.");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout() ==
                                    attributes.output_mem_config.memory_layout(),
                                "Error: Memory layout mismatch.");
                        }
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                            "Error: Shard orientation must be ROW_MAJOR.");
                        const auto M = operations::matmul::utilities::get_M_dim(
                            a_shape_padded, in0_tile, program_config.fuse_batch);
                        const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;

                        // No padding
                        TT_FATAL(M == per_core_M, "Error: M ({}) must be equal to per_core_M ({}).", M, per_core_M);
                        TT_FATAL(
                            per_core_M == (shard_shape[0] / in0_tile.get_height()),
                            "Error: per_core_M must be equal to shard_shape[0] ({}) / in0_tile.get_height() ({}).",
                            shard_shape[0],
                            in0_tile.get_height());
                        TT_FATAL(
                            K % program_config.in0_block_w == 0,
                            "Error: K {} must be divisible by in0_block_w {}.",
                            K,
                            program_config.in0_block_w);
                        if (!program_config.gather_in0) {  // Padding allowed for gather_in0
                            TT_FATAL(
                                (shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
                                "Error: shard_shape[1] ({}) / in0_tile.get_width() ({}) must be divisible by "
                                "in0_block_w ({}).",
                                shard_shape[1],
                                in0_tile.get_width(),
                                program_config.in0_block_w);
                        }
                    }
                    if (attributes.output_mem_config.is_sharded()) {
                        TT_FATAL(
                            attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                            "Error: Output memory layout must be WIDTH_SHARDED. Provided tensor memory layout: {}",
                            attributes.output_mem_config.memory_layout());
                        const auto M = operations::matmul::utilities::get_M_dim(
                            a_shape_padded, in0_tile, program_config.fuse_batch);
                        uint32_t per_core_M = program_config.per_core_M;
                        uint32_t per_core_N = program_config.per_core_N;

                        // No padding
                        TT_FATAL(M == per_core_M, "Error: M {} must be equal to per_core_M {}.", M, per_core_M);

                        TT_FATAL(
                            program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                            "Error: out_subblock_w must be equal to per_core_N or out_subblock_h must be equal to 1.");
                        TT_FATAL(
                            program_config.out_block_w == per_core_N || program_config.out_block_h == 1,
                            "Error: out_block_w must be equal to per_core_N or out_block_h must be equal to 1.");
                    }
                    if (input_tensor_b.buffer()->buffer_type() == tt_metal::BufferType::L1 &&
                        input_tensor_b.memory_config().is_sharded()) {
                        TT_FATAL(
                            input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                            "Operand B can only be interleaved or L1 width sharded.");
                        TT_FATAL(
                            program_config.per_core_N ==
                                (input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width()),
                            "Shard width must match per core N.");
                        if (optional_bias.has_value()) {
                            TT_FATAL(
                                input_tensor_b.shard_spec().value().shape[1] ==
                                    optional_bias.value().shard_spec().value().shape[1],
                                "Bias shard spec width must match second inputs shard spec "
                                "width.");
                        }
                    }
                } else {
                    if (input_tensor_a.memory_config().is_sharded()) {
                        TT_FATAL(program_config.fuse_batch, "Error: Batch fusion must be enabled.");
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                            "Error: input_tensor_a must be height sharded.");
                        if (attributes.output_mem_config.is_sharded()) {
                            TT_FATAL(
                                input_tensor_a.memory_config().buffer_type() ==
                                    attributes.output_mem_config.buffer_type(),
                                "Error: Buffer type mismatch.");
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout() ==
                                    attributes.output_mem_config.memory_layout(),
                                "Error: Memory layout mismatch.");
                        }
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                            "Error: Shard orientation must be ROW_MAJOR.");
                        const auto M = operations::matmul::utilities::get_M_dim(
                            a_shape_padded, in0_tile, program_config.fuse_batch);
                        const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
                        uint32_t per_core_M = program_config.per_core_M;
                        auto shard_shape = input_tensor_a.shard_spec().value().shape;
                        TT_FATAL(
                            div_up(M, per_core_M) <= input_tensor_a.shard_spec().value().grid.num_cores(),
                            "Error: M must be divisible by per_core_M.");
                        TT_FATAL(
                            per_core_M == (shard_shape[0] / in0_tile.get_height()),
                            "Error: per_core_M must be equal to shard_shape[0] / in0_tile.get_height().");
                        TT_FATAL(K % program_config.in0_block_w == 0, "Error: K must be divisible by in0_block_w.");
                        TT_FATAL(
                            K == (shard_shape[1] / in0_tile.get_width()),
                            "Error: K must be equal to shard_shape[1] / in0_tile.get_width().");
                    }
                    if (attributes.output_mem_config.is_sharded()) {
                        TT_FATAL(
                            attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                            "Error: Output memory layout must be HEIGHT_SHARDED.");
                        const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                        uint32_t per_core_N = program_config.per_core_N;

                        TT_FATAL(N == per_core_N, "Error: N must be equal to per_core_N.");
                        TT_FATAL(
                            program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                            "Error: out_subblock_w must be equal to per_core_N or out_subblock_h must be equal to 1.");
                        TT_FATAL(
                            program_config.out_block_w == per_core_N || program_config.out_block_h == 1,
                            "Error: out_block_w must be equal to per_core_N or out_block_h must be equal to 1.");
                    }
                    TT_FATAL(
                        input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                        "Error: Operand B must be interleaved.");
                }
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig>) {
                TT_FATAL(input_tensor_a.is_sharded(), "Input tensor A must be sharded for DRAM sharded program config");
                TT_FATAL(
                    attributes.output_mem_config.is_sharded(),
                    "Output memory config must be sharded for DRAM sharded program config");
                TT_FATAL(
                    input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                    "Input A memory layout must be WIDTH_SHARDED, got: {}",
                    input_tensor_a.memory_config().memory_layout());
                TT_FATAL(
                    input_tensor_a.memory_config().buffer_type() == attributes.output_mem_config.buffer_type(),
                    "Input A and output buffer types must match, got input: {} vs output: {}",
                    input_tensor_a.memory_config().buffer_type(),
                    attributes.output_mem_config.buffer_type());
                TT_FATAL(
                    input_tensor_a.memory_config().memory_layout() == attributes.output_mem_config.memory_layout(),
                    "Input A and output memory layouts must match, got input: {} vs output: {}",
                    input_tensor_a.memory_config().memory_layout(),
                    attributes.output_mem_config.memory_layout());
                TT_FATAL(
                    input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                    "Input A shard orientation must be ROW_MAJOR, got: {}",
                    input_tensor_a.shard_spec().value().orientation);
                const auto M = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
                const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
                uint32_t per_core_M = program_config.per_core_M;
                auto shard_shape = input_tensor_a.shard_spec().value().shape;

                // No padding
                TT_FATAL(M == per_core_M, "M ({}) must equal per_core_M ({})", M, per_core_M);
                TT_FATAL(M == 1, "currently only support in0 tensor height of tile height");
                TT_FATAL(
                    per_core_M == (shard_shape[0] / in0_tile.get_height()),
                    "per_core_M ({}) must equal shard_shape[0] / in0_tile.get_height() ({})",
                    per_core_M,
                    (shard_shape[0] / in0_tile.get_height()));
                TT_FATAL(
                    K % program_config.in0_block_w == 0,
                    "K ({}) must be divisible by in0_block_w ({})",
                    K,
                    program_config.in0_block_w);
                TT_FATAL(
                    (shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
                    "shard_shape[1] / in0_tile.get_width() ({}) must be divisible by in0_block_w ({})",
                    (shard_shape[1] / in0_tile.get_width()),
                    program_config.in0_block_w);

                // tensor in1
                TT_FATAL(
                    input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED,
                    "Input B memory layout must be WIDTH_SHARDED, got: {}",
                    input_tensor_b.memory_config().memory_layout());
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig>) {
                check_tensor_in_grid(input_tensor_a, program_config.compute_with_storage_grid_size);
                check_tensor_in_grid(input_tensor_b, program_config.compute_with_storage_grid_size);
                TT_FATAL(
                    program_config.per_core_M % program_config.out_block_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_M,
                    program_config.out_block_h);
                TT_FATAL(
                    program_config.per_core_N % program_config.out_block_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.per_core_N,
                    program_config.out_block_w);
                TT_FATAL(
                    program_config.out_block_h % program_config.out_subblock_h == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_h,
                    program_config.out_subblock_h);
                TT_FATAL(
                    program_config.out_block_w % program_config.out_subblock_w == 0,
                    "Error: incompatible values {} and {}",
                    program_config.out_block_w,
                    program_config.out_subblock_w);
                if (input_tensor_a.memory_config().is_sharded()) {
                    TT_FATAL(program_config.fuse_batch, "Batch fusion is required when input A is sharded");
                    auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
                    const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, in0_tile);
                    uint32_t per_core_M = program_config.per_core_M;
                    auto shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(
                        tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED ||
                            tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
                        "Unsupported memory layout {}.",
                        tensor_a_memory_layout);

                    if (tensor_a_memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                        if (program_config.transpose_mcast) {
                            TT_FATAL(
                                input_tensor_a.shard_spec().value().orientation == ShardOrientation::COL_MAJOR,
                                "Input tensor A must have COL_MAJOR shard orientation for transpose MCAST, got: {}",
                                input_tensor_a.shard_spec().value().orientation);
                        } else {
                            TT_FATAL(
                                input_tensor_a.shard_spec().value().orientation == ShardOrientation::ROW_MAJOR,
                                "Input tensor A must have ROW_MAJOR shard orientation for non-transpose MCAST, got: {}",
                                input_tensor_a.shard_spec().value().orientation);
                        }
                        if (attributes.output_mem_config.is_sharded()) {
                            TT_FATAL(
                                input_tensor_a.memory_config().buffer_type() ==
                                    attributes.output_mem_config.buffer_type(),
                                "Input tensor A and output buffer types must match, got input: {} vs output: {}",
                                input_tensor_a.memory_config().buffer_type(),
                                attributes.output_mem_config.buffer_type());
                            TT_FATAL(
                                input_tensor_a.memory_config().memory_layout() ==
                                    attributes.output_mem_config.memory_layout(),
                                "Input tensor A and output memory layouts must match, got input: {} vs output: {}",
                                input_tensor_a.memory_config().memory_layout(),
                                attributes.output_mem_config.memory_layout());
                        }

                    } else if (tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                        TT_FATAL(
                            !program_config.transpose_mcast,
                            "Transpose MCAST not supported with HEIGHT_SHARDED layout");
                        TT_FATAL(
                            K == program_config.in0_block_w,
                            "K ({}) must equal in0_block_w ({})",
                            K,
                            program_config.in0_block_w);
                        TT_FATAL(
                            program_config.in0_block_w == (shard_shape[1] / in0_tile.get_width()),
                            "in0_block_w ({}) must equal shard_shape[1] / in0_tile.get_width() ({})",
                            program_config.in0_block_w,
                            (shard_shape[1] / in0_tile.get_width()));
                        TT_FATAL(
                            input_tensor_a.shard_spec()->grid.bounding_box().start_coord.x ==
                                input_tensor_a.shard_spec()->grid.bounding_box().end_coord.x,
                            "Grid bounding box x coordinates must match, got start: {} vs end: {}",
                            input_tensor_a.shard_spec()->grid.bounding_box().start_coord.x,
                            input_tensor_a.shard_spec()->grid.bounding_box().end_coord.x);
                    }

                    TT_FATAL(
                        per_core_M == (shard_shape[0] / in0_tile.get_height()),
                        "per_core_M ({}) must equal shard_shape[0] / in0_tile.get_height() ({})",
                        per_core_M,
                        (shard_shape[0] / in0_tile.get_height()));
                    TT_FATAL(
                        (shard_shape[1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
                        "shard_shape[1] / in0_tile.get_width() ({}) must be divisible by in0_block_w ({})",
                        (shard_shape[1] / in0_tile.get_width()),
                        program_config.in0_block_w);
                }

                if (input_tensor_b.memory_config().is_sharded()) {
                    TT_FATAL(!program_config.transpose_mcast, "Transpose MCAST not supported when input B is sharded");
                    auto tensor_b_memory_layout = input_tensor_b.memory_config().memory_layout();
                    TT_FATAL(
                        tensor_b_memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
                        "Input B memory layout must be WIDTH_SHARDED, got: {}",
                        tensor_b_memory_layout);
                    if (input_tensor_b.buffer()->buffer_type() != tt_metal::BufferType::DRAM) {
                        const auto tensor_a_memory_layout = input_tensor_a.memory_config().memory_layout();
                        TT_FATAL(
                            (input_tensor_a.memory_config().is_sharded() &&
                             tensor_a_memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) ||
                                tensor_a_memory_layout == TensorMemoryLayout::INTERLEAVED,
                            "Error - non-DRAM width sharded input B requires input A to be interleaved or height "
                            "sharded, rather than {}",
                            tensor_a_memory_layout);
                        TT_FATAL(
                            program_config.per_core_N ==
                                (input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width()),
                            "per_core_N ({}) must equal input tensor B shard shape[1] / in1_tile.get_width() ({})",
                            program_config.per_core_N,
                            (input_tensor_b.shard_spec().value().shape[1] / in1_tile.get_width()));
                    }
                    TT_FATAL(
                        input_tensor_b.shard_spec()->grid.bounding_box().start_coord.y ==
                            input_tensor_b.shard_spec()->grid.bounding_box().end_coord.y,
                        "Input tensor B grid bounding box must have equal start and end y coordinates, got start: {} "
                        "vs end: {}",
                        input_tensor_b.shard_spec()->grid.bounding_box().start_coord.y,
                        input_tensor_b.shard_spec()->grid.bounding_box().end_coord.y);
                }

                if (attributes.output_mem_config.is_sharded()) {
                    TT_FATAL(
                        attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED,
                        "Output memory layout must be BLOCK_SHARDED, got: {}",
                        attributes.output_mem_config.memory_layout());
                    uint32_t per_core_N = program_config.per_core_N;

                    TT_FATAL(
                        program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                        "Error: out_subblock_w must be equal to per_core_N or out_subblock_h must be equal to 1.");
                    TT_FATAL(
                        program_config.out_block_w == per_core_N || program_config.out_block_h == 1,
                        "Error: out_block_w must be equal to per_core_N or out_block_h must be equal to 1.");
                }
            } else if constexpr (std::is_same_v<
                                     ProgramConfigType,
                                     operations::matmul::MatmulMultiCoreReuseProgramConfig>) {
                const auto M = operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/false);
                const auto total_M =
                    operations::matmul::utilities::get_M_dim(a_shape_padded, in0_tile, /*fuse_batch=*/true);
                const auto N = operations::matmul::utilities::get_N_dim(b_shape_padded, in1_tile);
                const auto K = operations::matmul::utilities::get_K_dim(a_shape_padded, /*tile=*/std::nullopt);
                uint32_t per_core_M = program_config.per_core_M;
                uint32_t per_core_N = program_config.per_core_N;
                if (per_core_M > M) {
                    TT_FATAL(
                        per_core_M % M == 0,
                        "per_core_M, {}, must be a multiple of M, {} if "
                        "per_core_M > M!",
                        per_core_M,
                        M);
                    TT_FATAL(
                        total_M % per_core_M == 0,
                        "input a total height, {}, must be divisible by "
                        "per_core_M, {}!",
                        total_M,
                        per_core_M);
                } else {
                    TT_FATAL(
                        M % per_core_M == 0, "per_core_M, {}, must divide M, {}, if per_core_M < M!", per_core_M, M);
                }
                TT_FATAL(N == per_core_N, "Error: N, {}, is not equal to per_core_N, {}", N, per_core_N);
                if (input_tensor_a.is_sharded()) {
                    TT_FATAL(
                        input_tensor_a.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                        "Error: memory layout, {}, is not width sharded",
                        input_tensor_a.memory_config().memory_layout());
                    auto in0_shard_shape = input_tensor_a.shard_spec().value().shape;

                    TT_FATAL(K == in0_shard_shape[1], "Error: K, {}, needs to be equal to {}", K, in0_shard_shape[1]);
                    TT_FATAL(
                        in0_shard_shape[1] == program_config.in0_block_w * in0_tile.get_width(),
                        "Error: {} needs to equal {} * {}",
                        in0_shard_shape[1],
                        program_config.in0_block_w,
                        in0_tile.get_width());
                    TT_FATAL(
                        per_core_M * in0_tile.get_height() == in0_shard_shape[0],
                        "Error: {} * {} needs to equal {}",
                        per_core_M,
                        in0_tile.get_height(),
                        in0_shard_shape[0]);

                    if (input_tensor_b.is_sharded()) {
                        TT_FATAL(
                            input_tensor_a.memory_config().buffer_type() ==
                                input_tensor_b.memory_config().buffer_type(),
                            "Input tensors A and B must have matching buffer types, got A: {} vs B: {}",
                            input_tensor_a.memory_config().buffer_type(),
                            input_tensor_b.memory_config().buffer_type());
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() ==
                                input_tensor_b.memory_config().memory_layout(),
                            "Input tensors A and B must have matching memory layouts, got A: {} vs B: {}",
                            input_tensor_a.memory_config().memory_layout(),
                            input_tensor_b.memory_config().memory_layout());
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().grid == input_tensor_b.shard_spec().value().grid,
                            "Input tensors A and B must have matching shard spec grids");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().orientation ==
                                input_tensor_b.shard_spec().value().orientation,
                            "Input tensors A and B must have matching shard orientations, got A: {} vs B: {}",
                            input_tensor_a.shard_spec().value().orientation,
                            input_tensor_b.shard_spec().value().orientation);
                    }
                    if (attributes.output_mem_config.is_sharded()) {
                        TT_FATAL(
                            input_tensor_a.memory_config().buffer_type() == attributes.output_mem_config.buffer_type(),
                            "Input tensor A and output buffer types must match, got input: {} vs output: {}",
                            input_tensor_a.memory_config().buffer_type(),
                            attributes.output_mem_config.buffer_type());
                        TT_FATAL(
                            input_tensor_a.memory_config().memory_layout() ==
                                attributes.output_mem_config.memory_layout(),
                            "Input tensor A and output memory layouts must match, got input: {} vs output: {}",
                            input_tensor_a.memory_config().memory_layout(),
                            attributes.output_mem_config.memory_layout());
                    }
                }

                const auto batch_size_a = get_batch_size(a_shape_padded);
                const auto batch_size_b = get_batch_size(b_shape_padded);
                bool broadcast_batch = batch_size_a > 1 and batch_size_b == 1;
                TT_FATAL(!broadcast_batch, "Batch broadcasting is not supported for the chosen program config");

                if (input_tensor_b.is_sharded()) {
                    TT_FATAL(per_core_M % M == 0, "per_core_M must be a multiple of M if input b is sharded!");
                    TT_FATAL(
                        input_tensor_b.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                        "Input B memory layout must not be WIDTH_SHARDED, got: {}",
                        input_tensor_b.memory_config().memory_layout());
                    auto in1_shard_shape = input_tensor_b.shard_spec().value().shape;
                    TT_FATAL(
                        in1_shard_shape[1] == b_shape_padded[-1],
                        "Input B shard shape[1] ({}) must equal padded shape[-1] ({})",
                        in1_shard_shape[1],
                        b_shape_padded[-1]);
                    TT_FATAL(
                        per_core_N * in1_tile.get_width() == in1_shard_shape[1],
                        "per_core_N * in1_tile.get_width() ({}) must equal in1_shard_shape[1] ({})",
                        per_core_N * in1_tile.get_width(),
                        in1_shard_shape[1]);
                    TT_FATAL(
                        in1_shard_shape[0] % K == 0,
                        "Input B shard shape[0] ({}) must be divisible by K ({})",
                        in1_shard_shape[0],
                        K);
                }
                if (attributes.output_mem_config.is_sharded()) {
                    TT_FATAL(
                        attributes.output_mem_config.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                        "Output memory layout must not be WIDTH_SHARDED, got: {}",
                        attributes.output_mem_config.memory_layout());
                    TT_FATAL(
                        program_config.out_subblock_w == per_core_N || program_config.out_subblock_h == 1,
                        "Either out_subblock_w ({}) must equal per_core_N ({}) or out_subblock_h ({}) must be 1",
                        program_config.out_subblock_w,
                        per_core_N,
                        program_config.out_subblock_h);
                }
            } else {
                TT_FATAL(
                    input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                    "Input A memory layout must be INTERLEAVED, got: {}",
                    input_tensor_a.memory_config().memory_layout());
                TT_FATAL(
                    input_tensor_b.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
                    "Input B memory layout must be INTERLEAVED, got: {}",
                    input_tensor_b.memory_config().memory_layout());
                TT_FATAL(
                    attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                    "Output memory layout must be INTERLEAVED, got: {}",
                    attributes.output_mem_config.memory_layout());
            }
            if constexpr (
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig> ||
                std::is_same_v<ProgramConfigType, operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig>) {
                TT_FATAL(
                    (a_shape_padded[-1] / in0_tile.get_width()) % program_config.in0_block_w == 0,
                    "Kt must be divisible by in0_block_w");
                TT_FATAL(
                    program_config.per_core_M % program_config.out_subblock_h == 0,
                    "per_core_M must be divisible by out_subblock_h");
                TT_FATAL(
                    program_config.per_core_N % program_config.out_subblock_w == 0,
                    "per_core_N must be divisible by out_subblock_w");
                uint32_t available_reg_count = ttnn::get_dest_reg_count(
                    attributes.compute_kernel_config.value(), attributes.output_tile.value().get_tile_shape());
                TT_FATAL(
                    (program_config.out_subblock_w * program_config.out_subblock_h) <= available_reg_count,
                    "out_subblock_w {} times out_subblock_h {} needs to be at "
                    "most {} to fit in hardware",
                    program_config.out_subblock_w,
                    program_config.out_subblock_h,
                    available_reg_count);
            }
        },
        chosen_program_config);
}

void MatmulDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    validate_on_program_cache_miss(attributes, args);
}

MatmulDeviceOperation::spec_return_value_t MatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    using namespace tt::tt_metal;
    using namespace tt::constants;
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

    TT_FATAL(attributes.output_dtype.has_value(), "Error: output_dtype field should have been populated");
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
                        "per_core_N must be divisible by override output tile width");
                    auto mem_config = attributes.output_mem_config;
                    if (!program_config.gather_in0) {
                        uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                        uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                        uint32_t num_cores = num_blocks_x * num_blocks_y;
                        CoreRangeSet all_cores =
                            num_cores_to_corerangeset(num_cores, program_config.compute_with_storage_grid_size, true);
                        tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec{
                            all_cores,
                            {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                            ShardOrientation::ROW_MAJOR};
                        mem_config = mem_config.with_shard_spec(shard_spec);
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
                        "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    auto grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
                    CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid_size, true);
                    ShardSpec shard_spec = ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                        ShardOrientation::ROW_MAJOR};
                    auto mem_config = attributes.output_mem_config.with_shard_spec(shard_spec);
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
                        "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                    CoreRangeSet all_cores;
                    ShardOrientation shard_orientation;
                    if (program_config.transpose_mcast) {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_y - 1, num_blocks_x - 1})});
                        shard_orientation = ShardOrientation::COL_MAJOR;
                    } else {
                        all_cores = CoreRangeSet({CoreRange({0, 0}, {num_blocks_x - 1, num_blocks_y - 1})});
                        shard_orientation = ShardOrientation::ROW_MAJOR;
                    }
                    tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                        shard_orientation};
                    auto mem_config = attributes.output_mem_config.with_shard_spec(shard_spec);
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
                        "per_core_N must be divisible by override output tile width");

                    uint32_t num_blocks_y = ((M - 1) / per_core_M) + 1;
                    uint32_t num_blocks_x = ((N - 1) / per_core_N) + 1;
                    uint32_t num_cores = num_blocks_x * num_blocks_y;
                    ShardOrientation shard_orientation = ShardOrientation::COL_MAJOR;
                    if (input_tensor_a.is_sharded()) {
                        shard_orientation = input_tensor_a.shard_spec().value().orientation;
                    } else if (input_tensor_b.is_sharded()) {
                        shard_orientation = input_tensor_b.shard_spec().value().orientation;
                    }

                    CoreRangeSet all_cores = num_cores_to_corerangeset(
                        num_cores,
                        program_config.compute_with_storage_grid_size,
                        shard_orientation == ShardOrientation::ROW_MAJOR);
                    tt::tt_metal::ShardSpec shard_spec = tt::tt_metal::ShardSpec{
                        all_cores,
                        {per_core_M * in0_tile.get_height(), per_core_N * in1_tile.get_width()},
                        shard_orientation};
                    auto mem_config = attributes.output_mem_config.with_shard_spec(shard_spec);
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

tt::stl::hash::hash_t MatmulDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& args) {
    const auto& input_tensors = args.input_tensors;
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    auto factory = select_program_factory(attributes, args);

    auto hash = tt::tt_metal::operation::hash_operation<MatmulDeviceOperation>(
        attributes, factory.index(), input_tensor_a, input_tensor_b);

    for (const auto& optional_input_tensor : args.optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            hash = tt::stl::hash::hash_objects(hash, optional_input_tensor.value());
        }
    }

    for (const auto& optional_output_tensor : args.optional_output_tensors) {
        if (optional_output_tensor.has_value()) {
            hash = tt::stl::hash::hash_objects(hash, optional_output_tensor.value());
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
    math_fidelity = are_inputs_32F ? MathFidelity::HiFi4 : math_fidelity;

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
    if (!attributes.program_config.has_value()) {
        uint32_t bias_single_tile_size = 0;
        if (bias.has_value()) {
            auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(bias.value().dtype());
            bias_single_tile_size = tt::tile_size(bias_data_format);
        }

        MatmulParams attributes_with_program_config = attributes;
        attributes_with_program_config.program_config = operations::matmul::get_program_config(
            input_tensor_a,
            input_tensor_b,
            attributes.transpose_a,
            attributes.transpose_b,
            bias_single_tile_size,
            attributes);

        return ttnn::device_operation::launch<MatmulDeviceOperation>(
            attributes_with_program_config, {{input_tensor_a, input_tensor_b}, {bias}, {optional_output_tensor}});
    }
    return ttnn::device_operation::launch<MatmulDeviceOperation>(
        attributes, {{input_tensor_a, input_tensor_b}, {bias}, {optional_output_tensor}});
}

MatmulDeviceOperation::tensor_return_value_t matmul(
    const std::vector<Tensor>& input_tensors,
    const std::optional<Tensor>& optional_output_tensor,
    const MatmulParams& attributes) {
    if (!attributes.program_config.has_value()) {
        uint32_t bias_single_tile_size = 0;

        MatmulParams attributes_with_program_config = attributes;
        attributes_with_program_config.program_config = operations::matmul::get_program_config(
            input_tensors.at(0),
            input_tensors.at(1),
            attributes.transpose_a,
            attributes.transpose_b,
            bias_single_tile_size,
            attributes);

        return ttnn::device_operation::launch<MatmulDeviceOperation>(
            attributes_with_program_config, {input_tensors, {}, {optional_output_tensor}});
    }

    return ttnn::device_operation::launch<MatmulDeviceOperation>(
        attributes, {input_tensors, {}, {optional_output_tensor}});
}

}  // namespace ttnn::prim

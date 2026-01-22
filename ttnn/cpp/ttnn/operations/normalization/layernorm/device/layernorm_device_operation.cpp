// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/constants.hpp>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

LayerNormDeviceOperation::program_factory_t LayerNormDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return LayerNormShardedProgramFactory{};
    }
    return LayerNormMultiCoreProgramFactory{};
}

void LayerNormDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void LayerNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& b = tensor_args.residual_input_tensor;
    const auto& gamma = tensor_args.weight;
    const auto& beta = tensor_args.bias;
    const auto& stats = tensor_args.stats;

    TT_FATAL(a.layout() == Layout::TILE, "Input tensor must have TILE layout, got: {}", a.layout());
    TT_FATAL(
        a.dtype() == DataType::FLOAT32 or a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B,
        "Input tensor must be FLOAT32, BFLOAT16, or BFLOAT8_B, got: {}",
        a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(
            b.value().layout() == Layout::TILE, "Residual tensor must have TILE layout, got: {}", b.value().layout());
        TT_FATAL(
            a.padded_shape() == b.value().padded_shape(),
            "Input and residual shapes must match, got input: {} vs residual: {}",
            a.padded_shape(),
            b.value().padded_shape());
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b.value().device(), "Input and residual tensors must be on same device");
    }

    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == gamma.value().padded_shape()[-1],
                "{} != {}",
                a.padded_shape()[-1],
                gamma.value().padded_shape()[-1]);
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().padded_shape()[-2] == TILE_HEIGHT,
                "Gamma tensor height must be TILE_HEIGHT (32), got: {}",
                gamma.value().padded_shape()[-2]);
        } else {
            TT_FATAL(
                gamma.value().layout() == Layout::ROW_MAJOR,
                "Gamma tensor must have ROW_MAJOR layout, got: {}",
                gamma.value().layout());
            TT_FATAL(
                (gamma.value().padded_shape()[-1] == TILE_WIDTH &&
                 gamma.value().physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Gamma's last padded dim needs to equal tile width and gamma's volume needs to align with last padded "
                "dim of input. Error with gamma.value().padded_shape(): {}, TILE_WIDTH: {}, "
                "gamma.value().physical_volume(): {}, "
                "a.padded_shape(): {}",
                gamma.value().padded_shape(),
                TILE_WIDTH,
                gamma.value().physical_volume(),
                a.padded_shape());
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().dtype() == DataType::FLOAT32 or gamma.value().dtype() == DataType::BFLOAT16,
                "Gamma tensor must be FLOAT32 or BFLOAT16, got: {}",
                gamma.value().dtype());
        }
        if (beta.has_value()) {
            TT_FATAL(gamma.value().layout() == beta.value().layout(), "Gamma and beta must have the same layout!");
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[-1] == beta.value().padded_shape()[-1],
                "Input and beta inner dimensions must match, got input: {} vs beta: {}",
                a.padded_shape()[-1],
                beta.value().padded_shape()[-1]);
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta.value().device(), "Input and beta tensors must be on same device");
            TT_FATAL(
                beta.value().padded_shape()[-2] == TILE_HEIGHT,
                "Beta tensor height must be TILE_HEIGHT (32), got: {}",
                beta.value().padded_shape()[-2]);
        } else {
            TT_FATAL(
                beta.value().layout() == Layout::ROW_MAJOR,
                "Beta tensor must have ROW_MAJOR layout, got: {}",
                beta.value().layout());
            TT_FATAL(
                (beta.value().padded_shape()[-1] == TILE_WIDTH &&
                 beta.value().physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Beta tensor dimensions must align with input tensor. Got beta padded shape: {}, physical volume: {}, "
                "input padded shape: {}, TILE_WIDTH: {}",
                beta.value().padded_shape()[-1],
                beta.value().physical_volume(),
                a.padded_shape()[-1],
                TILE_WIDTH);
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta.value().device(), "Input and beta tensors must be on same device");
            TT_FATAL(
                beta.value().dtype() == DataType::FLOAT32 or beta.value().dtype() == DataType::BFLOAT16,
                "Beta tensor must be FLOAT32 or BFLOAT16, got: {}",
                beta.value().dtype());
        }
    }
    if (a.is_sharded()) {
        // TODO: Add support for this (should be similar to interleaved)
        TT_FATAL(
            a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Height sharded inputs are not supported.");
        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded() &&
                operation_attributes.output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Sharded inputs require sharded outputs.");
        if (b.has_value()) {
            TT_FATAL(b.value().is_sharded(), "residual tensor b should be sharded if input a is sharded");
            TT_FATAL(b.value().shard_spec() == a.shard_spec(), "Both a and b should have the same shard spec");
            TT_FATAL(b.value().memory_config() == a.memory_config(), "Both a and b should have the same memory config");
        }
    }
    if (operation_attributes.distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER ||
        operation_attributes.distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
        TT_FATAL(a.padded_shape()[-2] == TILE_HEIGHT, "Only activations with batch size = 32 are supported");
        if (b.has_value()) {
            TT_FATAL(
                b.value().padded_shape()[-2] == TILE_HEIGHT,
                "Only residual tensors with batch size = 32 are supported");
        }
    }
    if (operation_attributes.distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
        TT_FATAL(stats.has_value(), "Post all gather layernorm requires stats");
        TT_FATAL(stats.value().is_sharded(), "Stats must be sharded");
        TT_FATAL(stats.value().layout() == Layout::TILE, "Only tile layout is supported for stats");
        TT_FATAL(stats.value().dtype() == DataType::BFLOAT16, "Only bfloat16 is supported for stats");
        TT_FATAL(stats.value().storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
        TT_FATAL(stats.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        if (operation_attributes.norm_type == LayerNormType::LAYERNORM) {
            TT_FATAL(
                stats.value().padded_shape()[-1] % (2 * TILE_WIDTH) == 0,
                "Stats is expected to have E(x) and E(x^2) for each device stacked interleaved in the last dimension");
        } else {
            TT_FATAL(
                stats.value().padded_shape()[-1] % TILE_WIDTH == 0,
                "Stats is expected to have E(x) for each device stacked in the last dimension");
        }
    }
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormDefaultProgramConfig>) {
                if (program_config.use_welford) {
                    TT_FATAL(
                        operation_attributes.norm_type != LayerNormType::RMSNORM,
                        "Welford's algorithm is not supported for RMSNorm");
                }
                if (operation_attributes.norm_type == LayerNormType::RMSNORM) {
                    TT_FATAL(!program_config.use_welford, "Welford's algorithm is not supported for RMSNorm");
                }
            } else if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                if (program_config.use_welford) {
                    TT_FATAL(
                        operation_attributes.norm_type != LayerNormType::RMSNORM,
                        "Welford's algorithm is not supported for RMSNorm");
                    TT_FATAL(
                        operation_attributes.distributed_norm_stage == DistributedLayerNormStage::NOT_DISTRIBUTED,
                        "Welford's algorithm is not supported for distributed layernorm");
                }
                if (program_config.inplace) {
                    TT_FATAL(
                        operation_attributes.output_mem_config.is_sharded(),
                        "Output memory config must be sharded for inplace operation");
                }
                TT_FATAL(
                    a.memory_config().buffer_type() == operation_attributes.output_mem_config.buffer_type(),
                    "Input and output buffer types must match, got input: {} vs output: {}",
                    a.memory_config().buffer_type(),
                    operation_attributes.output_mem_config.buffer_type());
                TT_FATAL(
                    a.memory_config().memory_layout() == operation_attributes.output_mem_config.memory_layout(),
                    "Input and output memory layouts must match, got input: {} vs output: {}",
                    a.memory_config().memory_layout(),
                    operation_attributes.output_mem_config.memory_layout());

                // tensor shape
                const auto& shape = a.padded_shape();
                uint32_t M = a.physical_volume() / shape[-1];
                uint32_t K = shape[-1];
                uint32_t Mt = M / TILE_WIDTH;
                uint32_t Kt = K / TILE_WIDTH;
                // block
                uint32_t block_h = program_config.block_h * TILE_HEIGHT;
                const auto shard_spec = a.shard_spec().value();
                // check dims
                TT_FATAL(
                    program_config.block_w % program_config.subblock_w == 0,
                    "block_w must be divisible by subblock_w.");
                TT_FATAL(M % TILE_HEIGHT == 0, "M ({}) must be divisible by tile height ({})", M, TILE_HEIGHT);
                TT_FATAL(K % TILE_WIDTH == 0, "K ({}) must be divisible by tile width ({})", K, TILE_WIDTH);
                const auto bbox = shard_spec.grid.bounding_box();
                TT_FATAL(
                    bbox.end_coord.x - bbox.start_coord.x < program_config.compute_with_storage_grid_size.x &&
                        bbox.end_coord.y - bbox.start_coord.y < program_config.compute_with_storage_grid_size.y,
                    "Bounding box dimensions must be smaller than compute grid size, got bbox x: {} to {}, y: {} to {} "
                    "vs grid size x: {}, y: {}",
                    bbox.start_coord.x,
                    bbox.end_coord.x,
                    bbox.start_coord.y,
                    bbox.end_coord.y,
                    program_config.compute_with_storage_grid_size.x,
                    program_config.compute_with_storage_grid_size.y);

                bool mcast_1d = M == block_h;
                bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
                if (mcast_1d) {
                    TT_FATAL(
                        tt::div_up(Kt, shard_spec.num_cores()) == program_config.block_w,
                        "block_w ({}) must equal to K (in tiles) / num_cores ({})",
                        program_config.block_w,
                        tt::div_up(Kt, shard_spec.num_cores()));
                    TT_FATAL(
                        Mt == program_config.block_h,
                        "block_h ({}) must equal to M (in tiles) ({})",
                        program_config.block_h,
                        Mt);
                    TT_FATAL(
                        a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
                        "Height sharded memory layout is not supported, got: {}",
                        a.memory_config().memory_layout());
                } else {
                    if (row_wise) {
                        TT_FATAL(
                            tt::div_up(Kt, (bbox.end_coord.x + 1)) == program_config.block_w,
                            "block_w ({}) must equal to K (in tiles) / num_cores_c ({})",
                            program_config.block_w,
                            tt::div_up(Kt, (bbox.end_coord.x + 1)));
                        TT_FATAL(
                            Mt / (bbox.end_coord.y + 1) == program_config.block_h,
                            "block_h ({}) must equal to M (in tiles)/ num_cores_r ({})",
                            program_config.block_h,
                            Mt / (bbox.end_coord.y + 1));
                    } else {
                        TT_FATAL(
                            tt::div_up(Kt, (bbox.end_coord.y + 1)) == program_config.block_w,
                            "block_w ({}) must equal to K (in tiles) / num_cores_r ({})",
                            program_config.block_w,
                            tt::div_up(Kt, (bbox.end_coord.y + 1)));
                        TT_FATAL(
                            Mt / (bbox.end_coord.x + 1) == program_config.block_h,
                            "block_h ({}) must equal to M (in tiles) / num_cores_c ({})",
                            program_config.block_h,
                            Mt / (bbox.end_coord.x + 1));
                    }
                }
                if (b.has_value()) {
                    TT_FATAL(b.value().is_sharded(), "Residual tensor must be sharded when input is sharded");
                    TT_FATAL(
                        b.value().shard_spec() == shard_spec,
                        "Residual tensor shard spec must match input shard spec, got residual: {} vs input: {}",
                        b.value().shard_spec(),
                        shard_spec);
                }
                TT_FATAL(
                    program_config.block_h * TILE_HEIGHT == shard_spec.shape[0],
                    "Block height * TILE_HEIGHT must match shard shape[0], got {} vs {}",
                    program_config.block_h * TILE_HEIGHT,
                    shard_spec.shape[0]);
                TT_FATAL(
                    program_config.block_w * TILE_WIDTH == shard_spec.shape[1],
                    "Block width * TILE_WIDTH must match shard shape[1], got {} vs {}",
                    program_config.block_w * TILE_WIDTH,
                    shard_spec.shape[1]);
                TT_FATAL(
                    program_config.block_w % program_config.subblock_w == 0,
                    "block_w must be divisible by subblock_w.");

                if (operation_attributes.distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
                    const auto stats_shard_spec = stats.value().shard_spec().value();
                    TT_FATAL(
                        stats_shard_spec.num_cores() == 1,
                        "Stats must be sharded with num_cores = 1, got: {}",
                        stats_shard_spec.num_cores());

                    if (operation_attributes.output_mem_config.shard_spec().has_value()) {
                        const auto output_shard_spec = operation_attributes.output_mem_config.shard_spec().value();
                        TT_FATAL(
                            output_shard_spec.shape[0] == shard_spec.shape[0],
                            "Output shard spec must have the same height as input shard spec, got output: {} vs input: "
                            "{}",
                            output_shard_spec.shape[0],
                            shard_spec.shape[0]);
                    }
                }
            }
        },
        operation_attributes.program_config);
}

TensorSpec LayerNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    if (operation_attributes.distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER) {
        uint32_t num_tiles_w = operation_attributes.norm_type == LayerNormType::LAYERNORM ? 2 : 1;
        output_shape[3] = num_tiles_w * TILE_WIDTH;
    }

    return std::visit(
        [&](const auto& program_config) -> spec_return_value_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                if (operation_attributes.distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER) {
                    auto shard_spec = input_tensor.shard_spec().value();
                    shard_spec.shape[1] = output_shape[3];
                    CoreCoord grid_start_core = shard_spec.grid.bounding_box().start_coord;
                    CoreRangeSet output_grid({CoreRange(grid_start_core, grid_start_core)});
                    shard_spec.grid = output_grid;
                    auto mem_config = operation_attributes.output_mem_config.with_shard_spec(shard_spec);
                    return TensorSpec(
                        output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_config));
                }
                if (operation_attributes.distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
                    auto output_shard_spec = operation_attributes.output_mem_config.shard_spec().value();
                    auto input_shard_spec = input_tensor.shard_spec().value();
                    if (output_shard_spec != input_shard_spec) {
                        output_padded_shape[3] = output_shard_spec.shape[1] * output_shard_spec.num_cores();
                    }
                }

                if (program_config.inplace) {
                    return input_tensor.tensor_spec();
                }

                auto mem_config = operation_attributes.output_mem_config;
                if (!mem_config.shard_spec().has_value()) {
                    mem_config = mem_config.with_shard_spec(input_tensor.shard_spec());
                }

                return ttnn::TensorSpec(
                    output_shape,
                    TensorLayout::fromPaddedShape(
                        operation_attributes.dtype.value_or(input_tensor.dtype()),
                        PageConfig(Layout::TILE),
                        mem_config,
                        output_shape,
                        output_padded_shape));
            } else {
                return TensorSpec(
                    output_shape,
                    TensorLayout(
                        input_tensor.dtype(), PageConfig(Layout::TILE), operation_attributes.output_mem_config));
            }
        },
        operation_attributes.program_config);
}

Tensor LayerNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return std::visit(
        [&](const auto& program_config) -> tensor_return_value_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                if (operation_attributes.distributed_norm_stage != DistributedLayerNormStage::PRE_ALL_GATHER &&
                    program_config.inplace) {
                    return tensor_args.input;
                }
            }
            auto output_spec = compute_output_specs(operation_attributes, tensor_args);
            return create_device_tensor(output_spec, tensor_args.input.device());
        },
        operation_attributes.program_config);
}

Tensor layer_norm(
    const Tensor& input_tensor,
    float epsilon,
    const std::optional<const Tensor>& weight,
    const std::optional<const Tensor>& bias,
    const std::optional<const Tensor>& residual_input_tensor,
    const MemoryConfig& output_mem_config,
    const LayerNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<DataType>& dtype,
    LayerNormType norm_type,
    DistributedLayerNormStage distributed_norm_stage,
    const std::optional<const Tensor>& stats) {
    auto operation_attributes = LayerNormParams{
        .norm_type = norm_type,
        .distributed_norm_stage = distributed_norm_stage,
        .eps = epsilon,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .dtype = dtype,
    };
    auto tensor_args = LayerNormInputs{
        .input = input_tensor,
        .residual_input_tensor = residual_input_tensor,
        .weight = weight,
        .bias = bias,
        .stats = stats,
    };

    return ttnn::device_operation::launch<LayerNormDeviceOperation>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim

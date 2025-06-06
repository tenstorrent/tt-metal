// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_op.hpp"
#include "layernorm_types.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/math.hpp"

#include <tt-metalium/constants.hpp>

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::normalization {

void LayerNorm::validate(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(
        input_tensors.size() == 1 and optional_input_tensors.size() <= 4, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    const auto& stats = optional_input_tensors.at(3);

    TT_FATAL(a.layout() == Layout::TILE, "Error");
    TT_FATAL(
        a.dtype() == DataType::FLOAT32 or a.dtype() == DataType::BFLOAT16 or a.dtype() == DataType::BFLOAT8_B, "Error");
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().layout() == Layout::TILE, "layot is not tile!");
        TT_FATAL(a.padded_shape() == b.value().padded_shape(), "shape is not same!");
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b.value().device(), "device is not same!");
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
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().padded_shape()[-2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(gamma.value().layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(
                (gamma.value().padded_shape()[-1] == TILE_WIDTH &&
                 gamma.value().physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Error");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(
                gamma.value().dtype() == DataType::FLOAT32 or gamma.value().dtype() == DataType::BFLOAT16, "Error");
        }
        if (beta.has_value()) {
            TT_FATAL(gamma.value().layout() == beta.value().layout(), "Gamma and beta must have the same layout!");
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(a.padded_shape()[-1] == beta.value().padded_shape()[-1], "Error");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(beta.value().padded_shape()[-2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(beta.value().layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL(
                (beta.value().padded_shape()[-1] == TILE_WIDTH &&
                 beta.value().physical_volume() / TILE_WIDTH == a.padded_shape()[-1] / TILE_WIDTH),
                "Error");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(beta.value().dtype() == DataType::FLOAT32 or beta.value().dtype() == DataType::BFLOAT16, "Error");
        }
    }
    if (a.is_sharded()) {
        // TODO: Add support for this (should be similar to interleaved)
        TT_FATAL(
            a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Hight sharded inputs are not supported.");
        TT_FATAL(
            this->output_mem_config.is_sharded() &&
                this->output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED,
            "Sharded inputs require sharded outputs.");
        if (b.has_value()) {
            TT_FATAL(b.value().is_sharded(), "residual tensor b should be sharded if input a is sharded");
            TT_FATAL(b.value().shard_spec() == a.shard_spec(), "Both a and b should have the same shard spec");
            TT_FATAL(b.value().memory_config() == a.memory_config(), "Both a and b should have the same memory config");
        }
    }
    if (this->distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER ||
        this->distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
        TT_FATAL(a.padded_shape()[-2] == TILE_HEIGHT, "Only activations with batch size = 32 are supported");
        if (b.has_value()) {
            TT_FATAL(
                b.value().padded_shape()[-2] == TILE_HEIGHT,
                "Only residual tensors with batch size = 32 are supported");
        }
    }
    if (this->distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
        TT_FATAL(stats.has_value(), "Post all gather layernorm requires stats");
        TT_FATAL(stats.value().is_sharded(), "Stats must be sharded");
        TT_FATAL(stats.value().layout() == Layout::TILE, "Only tile layout is supported for stats");
        TT_FATAL(stats.value().dtype() == DataType::BFLOAT16, "Only bfloat16 is supported for stats");
        TT_FATAL(stats.value().storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
        TT_FATAL(stats.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        if (this->norm_type == LayerNormType::LAYERNORM) {
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
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                if (program_config.inplace) {
                    TT_FATAL(this->output_mem_config.is_sharded(), "Error");
                }
                TT_FATAL(a.memory_config().buffer_type() == this->output_mem_config.buffer_type(), "Error");
                TT_FATAL(a.memory_config().memory_layout() == this->output_mem_config.memory_layout(), "Error");

                // tensor shape
                const auto shape = a.padded_shape();
                uint32_t M = a.physical_volume() / shape[-1];
                uint32_t K = shape[-1];
                uint32_t Mt = M / TILE_WIDTH;
                uint32_t Kt = K / TILE_WIDTH;
                // block
                uint32_t block_w = program_config.block_w * TILE_WIDTH;
                uint32_t block_h = program_config.block_h * TILE_HEIGHT;
                const auto shard_spec = a.shard_spec().value();
                uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;
                // check dims
                TT_FATAL(
                    program_config.block_w % program_config.subblock_w == 0,
                    "block_w must be divisible by subblock_w.");
                TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by tile height.");
                TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by tile width.");
                const auto bbox = shard_spec.grid.bounding_box();
                TT_FATAL(
                    bbox.end_coord.x - bbox.start_coord.x < program_config.compute_with_storage_grid_size.x &&
                        bbox.end_coord.y - bbox.start_coord.y < program_config.compute_with_storage_grid_size.y,
                    "Error");

                bool mcast_1d = M == block_h;
                bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
                if (mcast_1d) {
                    TT_FATAL(
                        tt::div_up(Kt, shard_spec.num_cores()) == program_config.block_w,
                        "block_w must equal to K / num_cores.");
                    TT_FATAL(Mt == program_config.block_h, "block_h must equal to M.");
                    TT_FATAL(a.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED, "Error");
                } else {
                    if (row_wise) {
                        TT_FATAL(
                            tt::div_up(Kt, (bbox.end_coord.x + 1)) == program_config.block_w,
                            "block_w must equal to K / num_cores_c.");
                        TT_FATAL(
                            Mt / (bbox.end_coord.y + 1) == program_config.block_h,
                            "block_h must equal to M / num_cores_r.");
                    } else {
                        TT_FATAL(
                            tt::div_up(Kt, (bbox.end_coord.y + 1)) == program_config.block_w,
                            "block_w must equal to K / num_cores_r.");
                        TT_FATAL(
                            Mt / (bbox.end_coord.x + 1) == program_config.block_h,
                            "block_h must equal to M / num_cores_c.");
                    }
                }
                if (b.has_value()) {
                    TT_FATAL(b.value().is_sharded(), "Error");
                    TT_FATAL(b.value().shard_spec() == shard_spec, "Error");
                }
                TT_FATAL(program_config.block_h * TILE_HEIGHT == shard_spec.shape[0], "Error");
                TT_FATAL(program_config.block_w * TILE_WIDTH == shard_spec.shape[1], "Error");
                TT_FATAL(
                    program_config.block_w % program_config.subblock_w == 0,
                    "block_w must be divisible by subblock_w.");

                if (this->distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
                    const auto stats_shard_spec = stats.value().shard_spec().value();
                    TT_FATAL(stats_shard_spec.num_cores() == 1, "Stats must be sharded with num_cores = 1");

                    if (this->output_mem_config.shard_spec().has_value()) {
                        const auto output_shard_spec = this->output_mem_config.shard_spec().value();
                        TT_FATAL(
                            output_shard_spec.shape[0] == shard_spec.shape[0],
                            "Output shard spec must have the same height as input shard spec.");
                    }
                }
            }
        },
        this->program_config);
}
std::vector<TensorSpec> LayerNorm::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    if (this->distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER) {
        uint32_t num_tiles_w = this->norm_type == LayerNormType::LAYERNORM ? 2 : 1;
        output_shape[3] = num_tiles_w * TILE_WIDTH;
    }

    return std::visit(
        [&](const auto& program_config) -> std::vector<TensorSpec> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                if (this->distributed_norm_stage == DistributedLayerNormStage::PRE_ALL_GATHER) {
                    auto shard_spec = input_tensor.shard_spec().value();
                    shard_spec.shape[1] = output_shape[3];
                    CoreCoord grid_start_core = shard_spec.grid.bounding_box().start_coord;
                    CoreRangeSet output_grid({CoreRange(grid_start_core, grid_start_core)});
                    shard_spec.grid = output_grid;
                    auto mem_config = this->output_mem_config.with_shard_spec(shard_spec);
                    return {TensorSpec(
                        output_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), mem_config))};
                } else if (this->distributed_norm_stage == DistributedLayerNormStage::POST_ALL_GATHER) {
                    auto output_shard_spec = this->output_mem_config.shard_spec().value();
                    auto input_shard_spec = input_tensor.shard_spec().value();
                    if (output_shard_spec != input_shard_spec) {
                        output_padded_shape[3] = output_shard_spec.shape[1] * output_shard_spec.num_cores();
                    }
                }

                if (program_config.inplace) {
                    return {input_tensor.tensor_spec()};
                }

                auto mem_config = this->output_mem_config;
                if (!mem_config.shard_spec().has_value()) {
                    mem_config = mem_config.with_shard_spec(input_tensor.shard_spec().value());
                }

                return {ttnn::TensorSpec(
                    output_shape,
                    TensorLayout::fromPaddedShape(
                        this->dtype.value_or(input_tensor.dtype()),
                        PageConfig(Layout::TILE),
                        mem_config,
                        output_shape,
                        output_padded_shape))};
            } else {
                return {TensorSpec(
                    output_shape,
                    TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), this->output_mem_config))};
            }
        },
        this->program_config);
}
std::vector<Tensor> LayerNorm::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return std::visit(
        [&](const auto& program_config) -> std::vector<Tensor> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                if (this->distributed_norm_stage != DistributedLayerNormStage::PRE_ALL_GATHER &&
                    program_config.inplace) {
                    return {input_tensors.at(0)};
                }
            }
            auto output_spec = compute_output_specs(input_tensors)[0];
            return {create_device_tensor(output_spec, input_tensors.at(0).device())};
        },
        this->program_config);
}
operation::ProgramWithCallbacks LayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    const auto& stats = optional_input_tensors.at(3);
    auto& output_tensor = output_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> tt::tt_metal::operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>) {
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);

                return layernorm_multi_core_sharded(
                    a,
                    b,
                    gamma,
                    beta,
                    stats,
                    output_tensor,
                    this->norm_type,
                    this->distributed_norm_stage,
                    this->eps,
                    program_config.compute_with_storage_grid_size,
                    program_config.subblock_w,
                    program_config.block_h,
                    program_config.block_w,
                    this->compute_kernel_config);
            } else {
                return layernorm_multi_core(
                    a, b, gamma, beta, output_tensor, this->norm_type, this->eps, this->compute_kernel_config);
            }
        },
        this->program_config);
}

}  // namespace ttnn::operations::normalization

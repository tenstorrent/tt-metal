// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "layernorm_op.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/math.hpp"

#include "tt_metal/common/constants.hpp"

#include <optional>

using uint32_t = std::uint32_t;
using namespace tt::constants;

namespace ttnn::operations::normalization {

void LayerNorm::validate(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<const Tensor>>& optional_input_tensors) const {
    TT_FATAL(input_tensors.size() == 1 and optional_input_tensors.size() <= 3, "Must have between 1 to 4 input tensors");
    auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    TT_FATAL(a.get_layout() == Layout::TILE, "Error");
    TT_FATAL(a.get_dtype() == DataType::FLOAT32 or a.get_dtype() == DataType::BFLOAT16 or a.get_dtype() == DataType::BFLOAT8_B, "Error");
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to layernorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");

    if (b.has_value()) {
        TT_FATAL(b.value().get_layout() == Layout::TILE, "layot is not tile!");
        TT_FATAL(a.get_legacy_shape() == b.value().get_legacy_shape(), "shape is not same!");
        TT_FATAL(b.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
        TT_FATAL(a.device() == b.value().device(), "device is not same!");
    }

    if (gamma.has_value()) {
        if (gamma.value().get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[-1] == gamma.value().get_legacy_shape()[-1], "{} != {}", a.get_legacy_shape()[-1], gamma.value().get_legacy_shape()[-1]);
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().get_legacy_shape()[-2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(gamma.value().get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL((gamma.value().get_legacy_shape()[-1] == TILE_WIDTH && gamma.value().volume() / TILE_WIDTH == a.get_legacy_shape()[-1] / TILE_WIDTH), "Error");
            TT_FATAL(gamma.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == gamma.value().device(), "Error");
            TT_FATAL(gamma.value().get_dtype() == DataType::FLOAT32 or gamma.value().get_dtype() == DataType::BFLOAT16, "Error");
        }
        if (beta.has_value()) {
            TT_FATAL(gamma.value().get_layout() == beta.value().get_layout(), "Gamma and beta must have the same layout!");
        }
    }

    if (beta.has_value()) {
        if (beta.value().get_layout() == Layout::TILE) {
            TT_FATAL(a.get_legacy_shape()[-1] == beta.value().get_legacy_shape()[-1], "Error");
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(beta.value().get_legacy_shape()[-2] == TILE_HEIGHT, "Error");
        } else {
            TT_FATAL(beta.value().get_layout() == Layout::ROW_MAJOR, "Error");
            TT_FATAL((beta.value().get_legacy_shape()[-1] == TILE_WIDTH && beta.value().volume() / TILE_WIDTH == a.get_legacy_shape()[-1] / TILE_WIDTH), "Error");
            TT_FATAL(beta.value().buffer() != nullptr, "Operands to layernorm need to be allocated in buffers on device!");
            TT_FATAL(a.device() == beta.value().device(), "Error");
            TT_FATAL(beta.value().get_dtype() == DataType::FLOAT32 or beta.value().get_dtype() == DataType::BFLOAT16, "Error");
        }
    }
    if (a.is_sharded()) {
        // TODO: Add support for this (should be similar to interleaved)
        TT_FATAL(a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED, "Error");
        TT_FATAL(this->output_mem_config.is_sharded() && this->output_mem_config.memory_layout != TensorMemoryLayout::HEIGHT_SHARDED, "Error");
    }
    std::visit(
        [&](const auto& program_config) {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>
            ) {
                if (program_config.inplace) {
                    TT_FATAL(this->output_mem_config.is_sharded(), "Error");
                }
                TT_FATAL(a.memory_config().buffer_type == this->output_mem_config.buffer_type, "Error");
                TT_FATAL(a.memory_config().memory_layout == this->output_mem_config.memory_layout, "Error");

                // tensor shape
                const auto shape = a.get_legacy_shape();
                uint32_t M = a.volume() / shape[-1];
                uint32_t K = shape[-1];
                uint32_t Mt = M / TILE_WIDTH;
                uint32_t Kt = K / TILE_WIDTH;
                // block
                uint32_t block_w = program_config.block_w * TILE_WIDTH;
                uint32_t block_h = program_config.block_h * TILE_HEIGHT;
                const auto shard_spec = a.shard_spec().value();
                uint32_t num_subblocks_w = program_config.block_w / program_config.subblock_w;
                // check dims
                TT_FATAL(program_config.block_w % program_config.subblock_w == 0, "block_w must be divisible by subblock_w.");
                TT_FATAL(M % TILE_HEIGHT == 0, "M must be divisible by tile height.");
                TT_FATAL(K % TILE_WIDTH == 0, "K must be divisible by tile width.");
                const auto bbox = shard_spec.grid.bounding_box();
                TT_FATAL(bbox.end_coord.x < program_config.compute_with_storage_grid_size.x && bbox.end_coord.y < program_config.compute_with_storage_grid_size.y, "Error");

                bool mcast_1d = M == block_h;
                bool row_wise = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
                if (mcast_1d) {
                    TT_FATAL(tt::div_up(Kt, shard_spec.num_cores()) == program_config.block_w, "block_w must equal to K / num_cores.");
                    TT_FATAL(Mt == program_config.block_h, "block_h must equal to M.");
                    TT_FATAL(a.memory_config().memory_layout != TensorMemoryLayout::HEIGHT_SHARDED, "Error");
                } else {
                    if (row_wise) {
                        TT_FATAL(tt::div_up(Kt, (bbox.end_coord.x + 1)) == program_config.block_w, "block_w must equal to K / num_cores_c.");
                        TT_FATAL(Mt / (bbox.end_coord.y + 1) == program_config.block_h, "block_h must equal to M / num_cores_r.");
                    } else {
                        TT_FATAL(tt::div_up(Kt, (bbox.end_coord.y + 1)) == program_config.block_w, "block_w must equal to K / num_cores_r.");
                        TT_FATAL(Mt / (bbox.end_coord.x + 1) == program_config.block_h, "block_h must equal to M / num_cores_c.");
                    }
                }
                if (b.has_value()) {
                    TT_FATAL(b.value().is_sharded(), "Error");
                    TT_FATAL(b.value().shard_spec() == shard_spec, "Error");
                }
                TT_FATAL(program_config.block_h * TILE_HEIGHT == shard_spec.shape[0], "Error");
                TT_FATAL(program_config.block_w * TILE_WIDTH == shard_spec.shape[1], "Error");
                TT_FATAL(program_config.block_w % program_config.subblock_w == 0, "block_w must be divisible by subblock_w.");
            }
        },
        this->program_config
    );


}
std::vector<tt::tt_metal::Shape> LayerNorm::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}
std::vector<Tensor> LayerNorm::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return std::visit(
        [&](const auto& program_config) -> std::vector<Tensor> {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>
            ) {
                if (program_config.inplace) {
                    return {input_tensor};
                } else {
                    auto mem_config = this->output_mem_config;
                    mem_config.shard_spec = input_tensor.shard_spec().value();
                    return {create_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensors.at(0).get_dtype(), Layout::TILE, input_tensor.device(), mem_config)};
                }
            } else {
                return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
            }
        },
        this->program_config
    );
}
operation::ProgramWithCallbacks LayerNorm::create_program(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor> &output_tensors
) const {
    const auto& a = input_tensors.at(0);
    const auto& b = optional_input_tensors.at(0);
    const auto& gamma = optional_input_tensors.at(1);
    const auto& beta = optional_input_tensors.at(2);
    auto& output_tensor = output_tensors.at(0);

    return std::visit(
        [&](const auto& program_config) -> operation::ProgramWithCallbacks {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if constexpr (
                std::is_same_v<ProgramConfigType, LayerNormShardedMultiCoreProgramConfig>
            ) {
                uint32_t num_cores_x = program_config.compute_with_storage_grid_size.x;
                uint32_t num_cores_y = program_config.compute_with_storage_grid_size.y;
                CoreCoord grid_size = CoreCoord(num_cores_x, num_cores_y);

                return layernorm_multi_core_sharded(
                                            a, b, gamma, beta, output_tensor, this->norm_type, this->eps,
                                            program_config.compute_with_storage_grid_size,
                                            program_config.subblock_w,
                                            program_config.block_h,
                                            program_config.block_w,
                                            this->compute_kernel_config
                                            );
            } else {
                return layernorm_multi_core(a, b, gamma, beta, output_tensor, this->norm_type, this->eps, this->compute_kernel_config);
            }
        },
        this->program_config
    );
}

}  // namespace ttnn::operations::normalization

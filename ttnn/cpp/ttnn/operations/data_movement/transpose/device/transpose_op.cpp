// SPDX-FileCopyrightText: © 2023 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_op.hpp"
#include "tt-metalium/assert.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "transpose_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void Transpose::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to transpose need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to transpose need to be allocated in buffers on device!");
    TT_FATAL(
        !(this->dim != TransposeOpDim::HC && this->pad_value.has_value() && this->pad_value != 0.0f),
        "Non-zero padding {} is not supported for any transpose other than HC.",
        this->pad_value.value());
    TT_FATAL(
        this->dim == TransposeOpDim::HC || this->dim == TransposeOpDim::WH || this->dim == TransposeOpDim::CN,
        "Transpose HC, WH, CN are the only supported transpose operations. Transpose {} is not supported.",
        (int)this->dim);
    const auto& shape = input_tensor.padded_shape();
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;
    uint32_t W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    uint32_t HW = H * W;
    if (not row_major) {
        TT_FATAL(
            W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0,
            "Tiled tensor H {} W {} must be a multiple of TILE HEIGHT {} and TILE WIDTH",
            H,
            W,
            TILE_HEIGHT,
            TILE_WIDTH);
        TT_FATAL(
            input_tensor.physical_volume() % TILE_HW == 0,
            "Tiled tensor volume {} must be a multiple of TILE HEIGHT * TILE WIDTH",
            input_tensor.physical_volume(),
            TILE_HW);
    }
    uint32_t ROW_MAJOR_STICK_WIDTH = 16;
    if (this->dim == TransposeOpDim::WH) {
        if (row_major) {
            TT_FATAL(
                (W * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0 &&
                    (H * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0,
                "Row major tensor W {} H {} must be a multiple of ROW_MAJOR_STICK_WIDTH for transpose wh",
                W,
                H,
                ROW_MAJOR_STICK_WIDTH);
        }
        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Only height and block sharding is supported for transpose wh");
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(
                (shard_spec.shape[0] % H == 0) || (H % shard_spec.shape[0] == 0),
                "Only a multiple of H {} or a factor of H is allows for the shard height {} for transpose WH",
                H,
                shard_spec.shape[0]);
            TT_FATAL(shard_spec.shape[1] == W, "Only height sharding is supported");
            if (H > shard_spec.shape[0]) {
                TT_FATAL(
                    N == 1,
                    "Transpose WH does not support sharded inputs when shard height {} is less than H {} and N {} > 1",
                    shard_spec.shape[0],
                    H,
                    N);
                TT_FATAL(
                    C == 1,
                    "Transpose WH does not support sharded inputs when  shard height {} is less than H {} and C {} > 1",
                    shard_spec.shape[0],
                    H,
                    N);
            }
            TT_FATAL(this->output_mem_config.is_sharded(), "Output must be sharded for transpose WH");
            TT_FATAL(
                this->output_mem_config.memory_layout() != TensorMemoryLayout::BLOCK_SHARDED,
                "Only height and width sharding output is supported for transpose wh");
        } else {
            TT_FATAL(!this->output_mem_config.is_sharded(), "Interleaved input tensors cannot output sharded outputs");
        }
    } else {
        if (input_tensor.is_sharded()) {
            TT_FATAL(
                input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Only height sharding is supported for transpose hc");
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(shard_spec.shape[1] == W, "Block/Width sharding is not supported");
            TT_FATAL(
                this->output_mem_config.is_sharded(), "Sharded input can only output sharded tensors for transpose hc");
            TT_FATAL(
                this->output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                "Only height sharding is supported for the ouput of sharded transpose hc");
        } else {
            TT_FATAL(!this->output_mem_config.is_sharded(), "Interleaved inputs cannot output sharded outputs");
        }
    }
    if (this->dim == TransposeOpDim::HC) {
        if (row_major) {
            auto BUFFER_ALIGNMENT = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                        ? hal::get_dram_alignment()
                                        : hal::get_l1_alignment();
            TT_FATAL(
                (W * input_tensor.element_size()) % BUFFER_ALIGNMENT == 0,
                "Buffer is not aligned for this implementation row_size_bytes {} buffer_alignment {}",
                W * input_tensor.element_size(),
                BUFFER_ALIGNMENT);
        }
        TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::INT32, "Unsupported data type for input tensor");
        TT_FATAL(
            !(input_tensor.is_sharded() && input_tensor.layout() == Layout::TILE),
            "HC transpose does not support sharded+tilized inputs");
        TT_FATAL(
            !(input_tensor.is_sharded() && pad_value.has_value() && pad_value.value() != 0.0f),
            "Sharded HC transpose does not support non-zero padding {}",
            pad_value.value());
    }
}

std::vector<ttnn::TensorSpec> Transpose::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);

    // TODO: Remove usage of input/output padded shape
    // - Get output alignment from input alignment and output dtype, layout, mem_config
    // - Get shard spec from output strides (logical shape + alignment)?
    auto output_shape = input_tensor.logical_shape();
    auto output_padded_shape = input_tensor.padded_shape();

    switch (this->dim) {
        case TransposeOpDim::CN:
            std::swap(output_shape[0], output_shape[1]);
            std::swap(output_padded_shape[0], output_padded_shape[1]);
            break;
        case TransposeOpDim::HC:
            if (input_tensor.is_sharded() || input_tensor.layout() != Layout::TILE) {
                std::swap(output_shape[1], output_shape[2]);
                std::swap(output_padded_shape[1], output_padded_shape[2]);
                break;
            } else {
                uint32_t C = output_shape[1];
                uint32_t C_p = tt::round_up(C, input_tensor.tensor_spec().tile().get_height());
                uint32_t H = output_shape[2];
                output_shape[1] = H;
                output_shape[2] = C;
                output_padded_shape[1] = H;
                output_padded_shape[2] = C_p;
                break;
            }

        case TransposeOpDim::WH:
            std::swap(output_shape[2], output_shape[3]);
            std::swap(output_padded_shape[2], output_padded_shape[3]);
            break;
        case TransposeOpDim::NH:
            std::swap(output_shape[0], output_shape[2]);
            std::swap(output_padded_shape[0], output_padded_shape[2]);
            break;
        case TransposeOpDim::NW:
            std::swap(output_shape[0], output_shape[3]);
            std::swap(output_padded_shape[0], output_padded_shape[3]);
            break;
        case TransposeOpDim::CW:
            std::swap(output_shape[1], output_shape[3]);
            std::swap(output_padded_shape[1], output_padded_shape[3]);
            break;
    }

    auto output_mem_config = this->output_mem_config;
    if (this->output_mem_config.is_sharded()) {
        TT_FATAL(input_tensor.is_sharded(), "Sharded output tensor must have a sharded input tensor");
        if (this->dim == TransposeOpDim::WH) {
            const auto& input_padded_shape = input_tensor.padded_shape();
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            if (shard_spec.shape[0] >= input_padded_shape[-2]) {
                shard_spec.shape[0] = shard_spec.shape[0] / input_padded_shape[-2] * input_padded_shape[-1];
                shard_spec.shape[1] = input_padded_shape[-2];
                output_mem_config = output_mem_config.with_shard_spec(shard_spec);
            } else {
                std::swap(shard_spec.shape[0], shard_spec.shape[1]);
                output_mem_config =
                    MemoryConfig(TensorMemoryLayout::WIDTH_SHARDED, output_mem_config.buffer_type(), shard_spec);
            }
        } else if (this->dim == TransposeOpDim::HC) {
            output_mem_config = output_mem_config.with_shard_spec(input_tensor.shard_spec().value());
        } else {
            TT_ASSERT(false, "Unsupported sharding");
        }
    }
    return {ttnn::TensorSpec(
        output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            output_mem_config,
            output_shape,
            output_padded_shape))};
}

operation::ProgramWithCallbacks Transpose::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            if (input_tensor.is_sharded()) {
                if (input_tensor.layout() == Layout::ROW_MAJOR) {
                    return detail::transpose_wh_multi_core_sharded_rm(input_tensor, output_tensor);
                } else {
                    return detail::transpose_wh_multi_core_sharded(input_tensor, output_tensor);
                }
            } else {
                return detail::transpose_wh_multi_core(input_tensor, output_tensor);
            }
            break;
        case TransposeOpParallelizationStrategy::MULTI_CORE_HC:
            if (input_tensor.is_sharded()) {
                return detail::transpose_hc_multi_core_sharded(input_tensor, output_tensor);
            } else {
                return detail::transpose_hc_multi_core(input_tensor, output_tensor, pad_value);
            }
        case TransposeOpParallelizationStrategy::MULTI_CORE_CN:
            return detail::transpose_cn_multi_core(input_tensor, output_tensor);
        default: TT_THROW("Unsupported parallelization strategy");
    }
}

TransposeOpParallelizationStrategy Transpose::get_parallelization_strategy(
    const std::vector<Tensor>& input_tensors) const {
    if (this->dim == TransposeOpDim::WH) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (this->dim == TransposeOpDim::HC) {  // Always true for legal shape until requirement on tile size IO is
                                                   // no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else if (this->dim == TransposeOpDim::CN) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_CN;
    } else {
        TT_THROW("Unsupported Transpose Dim");
    }
}

}  // namespace ttnn::operations::data_movement

// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "transpose_op.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "transpose_program_factory.hpp"
using namespace tt::constants;


namespace ttnn::operations::data_movement {

void Transpose::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to transpose need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr , "Operands to transpose need to be allocated in buffers on device!");
    const auto shape = input_tensor.get_shape().with_tile_padding();
    bool row_major = input_tensor.get_layout() == Layout::ROW_MAJOR;
    uint32_t W = shape[3], H = shape[2], C = shape[1], N = shape[0];
    uint32_t HW = H*W;
    if (not row_major) {
        TT_FATAL(W % TILE_WIDTH == 0 && H % TILE_HEIGHT == 0, "Error");
        TT_FATAL(input_tensor.volume() % TILE_HW == 0, "Error");
    }
    uint32_t ROW_MAJOR_STICK_WIDTH = 16;
    if (this->dim == TransposeOpDim::WH) {
        if (row_major) {
            TT_FATAL((W * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0 && (H * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0, "Error");
        }
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(shard_spec.shape[1] == W, "Error");
            TT_FATAL(shard_spec.shape[0] % H == 0, "Error");
            TT_FATAL(this->output_mem_config.is_sharded(), "Error");
            TT_FATAL(this->output_mem_config.memory_layout != TensorMemoryLayout::WIDTH_SHARDED, "Error");
        } else {
            TT_FATAL(!this->output_mem_config.is_sharded(), "Error");
        }
    } else {
        if (input_tensor.is_sharded()) {
            TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
            const auto shard_spec = input_tensor.shard_spec().value();
            TT_FATAL(shard_spec.shape[1] == W, "Error");
            TT_FATAL(this->output_mem_config.is_sharded(), "Error");
            TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::HEIGHT_SHARDED, "Error");
        } else {
            TT_FATAL(!this->output_mem_config.is_sharded(), "Error");
        }
    }
    if (this->dim == TransposeOpDim::HC) {
        if (row_major) {
            TT_FATAL((W * input_tensor.element_size()) % ROW_MAJOR_STICK_WIDTH == 0, "Error");
        } else {
            TT_FATAL(C % TILE_HEIGHT == 0, "Error");
        }
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32, "Error");
    } else if (this->dim == TransposeOpDim::CW) {
        TT_FATAL(C % TILE_WIDTH == 0, "Error");
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32, "Error");
    } else if (this->dim == TransposeOpDim::NH) {
        TT_FATAL(N % TILE_HEIGHT == 0, "Error");
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32, "Error");
    } else if (this->dim == TransposeOpDim::NW) {
        TT_FATAL(N % TILE_WIDTH == 0, "Error");
        TT_FATAL(input_tensor.get_dtype() == DataType::BFLOAT16 || input_tensor.get_dtype() == DataType::FLOAT32, "Error");
    }
}


std::vector<ttnn::Shape> Transpose::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto out_shape = input_tensor.get_shape().with_tile_padding();
    auto padding = out_shape.padding();
    switch (this->dim){
        case TransposeOpDim::CN:
            std::swap(out_shape[0], out_shape[1]);
            std::swap(padding[0], padding[1]);
            break;
        case TransposeOpDim::HC:
            std::swap(out_shape[1], out_shape[2]);
            std::swap(padding[1], padding[2]);
            break;
        case TransposeOpDim::WH:
            std::swap(out_shape[2], out_shape[3]);
            std::swap(padding[2], padding[3]);
            break;
        case TransposeOpDim::NH:
            std::swap(out_shape[0], out_shape[2]);
            std::swap(padding[0], padding[2]);
            break;
        case TransposeOpDim::NW:
            std::swap(out_shape[0], out_shape[3]);
            std::swap(padding[0], padding[3]);
            break;
        case TransposeOpDim::CW:
            std::swap(out_shape[1], out_shape[3]);
            std::swap(padding[1], padding[3]);
            break;
    }
    return {ttnn::Shape(out_shape, padding)};
}


std::vector<Tensor> Transpose::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    // This is only for WH
    if (this->output_mem_config.is_sharded()) {
        if (this->dim == TransposeOpDim::WH) {
            ShardSpec shard_spec = input_tensor.shard_spec().value();
            shard_spec.shape[0] = shard_spec.shape[0] / input_tensor.get_shape().with_tile_padding()[-2] * input_tensor.get_shape().with_tile_padding()[-1];
            shard_spec.shape[1] = input_tensor.get_shape().with_tile_padding()[-2];
            const auto output_shape = this->compute_output_shapes(input_tensors)[0];
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = shard_spec;
            return {create_device_tensor(
                output_shape,
                input_tensor.get_dtype(),
                input_tensor.get_layout(),
                input_tensor.device(),
                mem_config)};
        } else if (this->dim == TransposeOpDim::HC) {
            const auto output_shape = this->compute_output_shapes(input_tensors)[0];
            auto mem_config = this->output_mem_config;
            mem_config.shard_spec = input_tensor.shard_spec().value();
            return {create_device_tensor(
                output_shape,
                input_tensor.get_dtype(),
                input_tensor.get_layout(),
                input_tensor.device(),
                mem_config)};
        } else {
            TT_ASSERT(false, "Unsupported sharding");
        }
    }
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks Transpose::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);

    switch (parallelization_strategy) {
        case TransposeOpParallelizationStrategy::MULTI_CORE_WH:
            if (input_tensor.is_sharded()) {
                if (input_tensor.get_layout() == Layout::ROW_MAJOR) {
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
                return detail::transpose_hc_multi_core(input_tensor, output_tensor);
            }
        case TransposeOpParallelizationStrategy::MULTI_CORE_CN:
            return detail::transpose_cn_multi_core(input_tensor, output_tensor);
        default:
            TT_THROW("Unsupported parallelization strategy");
    }
}

TransposeOpParallelizationStrategy Transpose::get_parallelization_strategy(const std::vector<Tensor>& input_tensors) const {
    if (this->dim == TransposeOpDim::WH) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_WH;
    } else if (this->dim == TransposeOpDim::HC) { // Always true for legal shape until requirement on tile size IO is no longer required
        return TransposeOpParallelizationStrategy::MULTI_CORE_HC;
    } else if (this->dim == TransposeOpDim::CN) {
        return TransposeOpParallelizationStrategy::MULTI_CORE_CN;
    } else {
        TT_THROW("Unsupported Transpose Dim");
    }
}




}

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/run_operation.hpp"

// using namespace tt;
// using namespace tt_metal;
// using namespace constants;

namespace ttnn::operations::data_movement {



operation::ProgramWithCallbacks bcast_multi_core_h(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &output_tensor, BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_sharded_h(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &output_tensor, BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_sharded_h_optimised(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &output_tensor, BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_multi_core_w(
    const Tensor &input_tensor_a, const Tensor &input_tensor_b, const Tensor &output_tensor, BcastOpMath bcast_op);
operation::ProgramWithCallbacks bcast_multi_core_hw(
    const Tensor &input_tensor_a,
    const Tensor &input_tensor_b,
    const Tensor &output_tensor,
    BcastOpMath bcast_op,
    bool inplace);


void EltwiseBinaryBroadcast::validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    TT_FATAL(input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr, "Operands to bcast need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.device() != nullptr and input_tensor_b.device() != nullptr, "Operands to bcast need to be on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to bcast need to be on the same device!");

    const auto input_shape_a = input_tensor_a.get_legacy_shape();
    const auto input_shape_b = input_tensor_b.get_legacy_shape();

    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE);
    TT_FATAL(input_tensor_b.get_layout() == Layout::TILE);
    TT_FATAL(is_floating_point(input_tensor_a.get_dtype()), "Unsupported data format");
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        TT_FATAL(is_floating_point(output_tensors.at(0).value().get_dtype()), "Unsupported data format");
        const std::vector<tt::tt_metal::LegacyShape> output_shape_required = this->compute_output_shapes(input_tensors);
        const auto& out_tensor = output_tensors.at(0).value();
        TT_FATAL(out_tensor.get_legacy_shape() == output_shape_required.at(0), fmt::format("The input tensors need a shape of {}, however the output tensor is only {}", output_shape_required,  out_tensor.get_legacy_shape()));
    }
    if (this->in_place) {
        TT_FATAL(input_tensor_a.memory_config().memory_layout == this->output_mem_config.memory_layout);
        TT_FATAL(input_tensor_a.memory_config().buffer_type == this->output_mem_config.buffer_type);
    }
    auto out_mem_config = (!output_tensors.empty() && output_tensors.at(0).has_value()) ? output_tensors.at(0).value().memory_config() : this->output_mem_config;
    if (this->dim == BcastOpDim::W){
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED &&
                out_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Bcast does not currently support input0 sharding, except if dim is W");
    } else if (this->dim == BcastOpDim::H) {
        if(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED){
            TT_FATAL(
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED &&
                out_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
            "Bcast does not currently support input0 sharding, except if dim is HW");
        }else{
            TT_FATAL(
            input_tensor_a.memory_config().is_sharded() && out_mem_config.is_sharded(),
            "Input and output mem layouts must be the same for bcast H op!");
        }
    } else {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED ||
                input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "HW bcast in0 supports Height Sharding or Interleaving");
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout == out_mem_config.memory_layout,
            "Input and output mem layouts must be the same for bcast HW op!");
    }

    auto height_a = input_shape_a[-2];
    auto width_a = input_shape_a[-1];
    auto height_b = input_shape_b[-2];
    auto width_b = input_shape_b[-1];
    if((input_tensor_a.is_sharded() && this->dim == BcastOpDim::H) == false){

        uint32_t batch_size_b = get_batch_size(input_shape_b);
        if (batch_size_b != 1) {
            TT_FATAL(input_shape_a.rank() == input_shape_b.rank() && "Broadcast with batch is currently only supported when input tensor ranks are the same");
            for (auto i = 0; i < input_shape_a.rank() - 2; i++) {
                TT_FATAL(
                        input_shape_a[i] == input_shape_b[i] &&
                        "Broadcast with batch is currently only supported when bN*bC=1 or N & C match or equivalent"); // for H multi-batch weight is supported
            }
        }
    }

    // validate input dimensions
    if (this->dim == BcastOpDim::W)
        TT_FATAL(height_a == height_b && width_b == TILE_WIDTH);
    if (this->dim == BcastOpDim::H)
        TT_FATAL(width_a == width_b && height_b == TILE_HEIGHT);
    if (this->dim == BcastOpDim::HW)
        TT_FATAL(width_b == TILE_WIDTH && height_b == TILE_HEIGHT);
}


std::vector<tt::tt_metal::LegacyShape> EltwiseBinaryBroadcast::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}


std::vector<Tensor> EltwiseBinaryBroadcast::create_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if(!output_tensors.empty() && output_tensors.at(0).has_value()){
        return {output_tensors.at(0).value()};
    }
    if (this->in_place) {
        return {};
    }
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        ShardSpec shard_spec{CoreRangeSet({}), {0, 0}};
        if (input_tensor.memory_config().is_sharded()) {
            // Derive output shard_spec based on input
            shard_spec = input_tensor.shard_spec().value();
        }
        auto mem_config = this->output_mem_config;
        mem_config.shard_spec = shard_spec;
        return {create_device_tensor(this->compute_output_shapes(input_tensors).at(0), input_tensor.get_dtype(), Layout::TILE, input_tensor.device(), mem_config)};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks EltwiseBinaryBroadcast::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto& output_tensor = this->in_place ? input_tensor_a : output_tensors.at(0);

    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    switch (parallelization_strategy){
        case BcastOpParallelizationStrategy::MULTI_CORE_H_SHARDED:
            return bcast_sharded_h(input_tensor_a, input_tensor_b, output_tensor, this->math_op);
        case BcastOpParallelizationStrategy::MULTI_CORE_H_SHARDED_OPTIMISED:
            return bcast_sharded_h_optimised(input_tensor_a, input_tensor_b, output_tensor, this->math_op);
        case BcastOpParallelizationStrategy::MULTI_CORE_H:
            return bcast_multi_core_h(input_tensor_a, input_tensor_b, output_tensor, this->math_op);
        case BcastOpParallelizationStrategy::MULTI_CORE_W:
            return bcast_multi_core_w(input_tensor_a, input_tensor_b, output_tensor, this->math_op);
        case BcastOpParallelizationStrategy::MULTI_CORE_HW:
            return bcast_multi_core_hw(input_tensor_a, input_tensor_b, output_tensor, this->math_op, this->in_place);
        default:
            TT_THROW("Unsupported Parallelization Strategy");
    }
}

const operation::Hash EltwiseBinaryBroadcast::compute_program_hash(
    const std::vector<Tensor> &input_tensors) const {
    auto parallelization_strategy = this->get_parallelization_strategy(input_tensors);
    bool bcast_scalar = (input_tensors.at(1).get_legacy_shape()[-2] * input_tensors.at(1).get_legacy_shape()[-1] == 1) && this->dim == BcastOpDim::HW;
    return operation::hash_operation<EltwiseBinaryBroadcast>(
        *this,
        parallelization_strategy,
        std::get<DeviceStorage>(input_tensors.at(0).storage()).memory_config(),
        input_tensors.at(0).dtype(),
        std::get<DeviceStorage>(input_tensors.at(1).storage()).memory_config(),
        input_tensors.at(1).dtype(),
        bcast_scalar,
        this->in_place);
}

BcastOpParallelizationStrategy EltwiseBinaryBroadcast::get_parallelization_strategy(const std::vector<Tensor> &input_tensors) const {
    using namespace tt::constants;
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    uint32_t num_tiles = input_tensor_a.volume() / TILE_HW;
    uint32_t Ht = input_tensor_a.get_legacy_shape()[-2] / TILE_HEIGHT;
    uint32_t Wt = input_tensor_a.get_legacy_shape()[-1] / TILE_WIDTH;

    if(this->dim == BcastOpDim::H){
        if(input_tensor_a.is_sharded())
            if (input_tensor_a.get_legacy_shape()[0] == input_tensor_b.get_legacy_shape()[0] || input_tensor_a.get_legacy_shape()[0] > 1 and input_tensor_b.get_legacy_shape()[0] == 1){
                return BcastOpParallelizationStrategy::MULTI_CORE_H_SHARDED_OPTIMISED;
            } else {
                return BcastOpParallelizationStrategy::MULTI_CORE_H_SHARDED;
            }
        else
            return BcastOpParallelizationStrategy::MULTI_CORE_H;
    }
    else if(this->dim == BcastOpDim::W){
        return BcastOpParallelizationStrategy::MULTI_CORE_W;
    }
    else if(this->dim == BcastOpDim::HW){
        return BcastOpParallelizationStrategy::MULTI_CORE_HW;
    }
    else{
        TT_THROW("Unsupported Bcast Dim");
    }
}

} // ttnn::operations::data_movement

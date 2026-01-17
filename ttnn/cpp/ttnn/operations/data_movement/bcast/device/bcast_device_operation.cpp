// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "bcast_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <tt-logger/tt-logger.hpp>
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement::bcast {

using namespace tt::constants;
using namespace tt::tt_metal;

BcastDeviceOperation::program_factory_t BcastDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor_a = tensor_args.input_a;
    const Tensor& input_tensor_b = tensor_args.input_b;

    program_factory_t selected_factory;
    [[maybe_unused]] const char* factory_name = nullptr;

    if (operation_attributes.dim == BcastOpDim::H) {
        if (input_tensor_a.is_sharded()) {
            if (input_tensor_a.padded_shape()[0] == input_tensor_b.padded_shape()[0] ||
                (input_tensor_a.padded_shape()[0] > 1 && input_tensor_b.padded_shape()[0] == 1)) {
                selected_factory = program::BcastShardedHOptimisedProgramFactory{};
                factory_name = "BcastShardedHOptimisedProgramFactory";
            } else {
                selected_factory = program::BcastShardedHProgramFactory{};
                factory_name = "BcastShardedHProgramFactory";
            }
        } else {
            selected_factory = program::BcastMultiCoreHProgramFactory{};
            factory_name = "BcastMultiCoreHProgramFactory";
        }
    } else if (operation_attributes.dim == BcastOpDim::W) {
        selected_factory = program::BcastMultiCoreWProgramFactory{};
        factory_name = "BcastMultiCoreWProgramFactory";
    } else if (operation_attributes.dim == BcastOpDim::HW) {
        selected_factory = program::BcastMultiCoreHWProgramFactory{};
        factory_name = "BcastMultiCoreHWProgramFactory";
    } else {
        TT_THROW("Unsupported Bcast Dim");
    }

    log_debug(tt::LogOp, "Selected bcast factory: {}", factory_name);
    return selected_factory;
}

void BcastDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void BcastDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const Tensor& input_tensor_a = tensor_args.input_a;
    const Tensor& input_tensor_b = tensor_args.input_b;

    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to bcast need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.device() != nullptr and input_tensor_b.device() != nullptr,
        "Operands to bcast need to be on device!");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Operands to bcast need to be on the same device!");

    const auto& input_shape_a = input_tensor_a.padded_shape();
    const auto& input_shape_b = input_tensor_b.padded_shape();

    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE,
        "Input tensor A layout must be TILE but got {}",
        input_tensor_a.layout());
    TT_FATAL(
        input_tensor_b.layout() == Layout::TILE,
        "Input tensor B layout must be TILE but got {}",
        input_tensor_b.layout());
    TT_FATAL(is_floating_point(input_tensor_a.dtype()), "Unsupported data format");
    if (tensor_args.preallocated_output.has_value()) {
        TT_FATAL(is_floating_point(tensor_args.preallocated_output->dtype()), "Unsupported data format");
        const auto output_spec_required = compute_output_specs(operation_attributes, tensor_args);
        const auto& out_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            out_tensor.logical_shape() == output_spec_required.logical_shape(),
            "The input tensors need a shape of {}, however the output tensor is only {}",
            output_spec_required.logical_shape(),
            out_tensor.padded_shape());
    }
    if (operation_attributes.in_place) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == operation_attributes.output_mem_config.memory_layout(),
            "Input tensor A memory layout ({}) must match output memory config layout ({})",
            input_tensor_a.memory_config().memory_layout(),
            operation_attributes.output_mem_config.memory_layout());
        TT_FATAL(
            input_tensor_a.memory_config().buffer_type() == operation_attributes.output_mem_config.buffer_type(),
            "Input tensor A buffer type ({}) must match output memory config buffer type ({})",
            input_tensor_a.memory_config().buffer_type(),
            operation_attributes.output_mem_config.buffer_type());
    }
    const MemoryConfig& out_mem_config = tensor_args.preallocated_output.has_value()
                                             ? tensor_args.preallocated_output->memory_config()
                                             : operation_attributes.output_mem_config;
    if (operation_attributes.dim == BcastOpDim::W) {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                out_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Bcast does not currently support input0 sharding, except if dim is W");
    } else if (operation_attributes.dim == BcastOpDim::H) {
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED) {
            TT_FATAL(
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
                    out_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                "Bcast does not currently support input0 sharding, except if dim is HW");
        } else {
            TT_FATAL(
                input_tensor_a.memory_config().is_sharded() && out_mem_config.is_sharded(),
                "Input and output mem layouts must be the same for bcast H op!");
        }
    } else {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED ||
                input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "HW bcast in0 supports Height Sharding or Interleaving");
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == out_mem_config.memory_layout(),
            "Input and output mem layouts must be the same for bcast HW op!");
    }

    const uint32_t height_a = input_shape_a[-2];
    const uint32_t width_a = input_shape_a[-1];
    const uint32_t height_b = input_shape_b[-2];
    const uint32_t width_b = input_shape_b[-1];
    if (!(input_tensor_a.is_sharded() && operation_attributes.dim == BcastOpDim::H)) {
        const uint32_t batch_size_b = get_batch_size(input_shape_b);
        if (batch_size_b != 1) {
            TT_FATAL(
                input_shape_a.rank() == input_shape_b.rank(),
                "Broadcast with batch is currently only supported when input tensor ranks are the same",
                "Error");
            for (auto i = 0; i < input_shape_a.rank() - 2; i++) {
                TT_FATAL(
                    input_shape_a[i] == input_shape_b[i],
                    "Broadcast with batch is currently only supported when bN*bC=1 or N & C match or equivalent");  // for H multi-batch weight is supported
            }
        }
    }

    // validate input dimensions
    if (operation_attributes.dim == BcastOpDim::W) {
        TT_FATAL(
            height_a == height_b && width_b == TILE_WIDTH,
            "For width broadcast: height_a ({}) must equal height_b ({}) and width_b ({}) must equal TILE_WIDTH ({})",
            height_a,
            height_b,
            width_b,
            TILE_WIDTH);
    }
    if (operation_attributes.dim == BcastOpDim::H) {
        TT_FATAL(
            width_a == width_b && height_b == TILE_HEIGHT,
            "For height broadcast: width_a ({}) must equal width_b ({}) and height_b ({}) must equal TILE_HEIGHT ({})",
            width_a,
            width_b,
            height_b,
            TILE_HEIGHT);
    }
    if (operation_attributes.dim == BcastOpDim::HW) {
        TT_FATAL(
            width_b == TILE_WIDTH && height_b == TILE_HEIGHT,
            "For HW broadcast: width_b ({}) must equal TILE_WIDTH ({}) and height_b ({}) must equal TILE_HEIGHT ({})",
            width_b,
            TILE_WIDTH,
            height_b,
            TILE_HEIGHT);
    }
}

TensorSpec BcastDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }
    if (operation_attributes.in_place) {
        return tensor_args.input_a.tensor_spec();
    }
    const Tensor& input_tensor = tensor_args.input_a;
    if (operation_attributes.output_mem_config.is_sharded()) {
        ShardSpec shard_spec{CoreRangeSet(), {0, 0}};
        if (input_tensor.memory_config().is_sharded()) {
            // Derive output shard_spec based on input
            shard_spec = input_tensor.shard_spec().value();
        }
        const MemoryConfig mem_config = operation_attributes.output_mem_config.with_shard_spec(shard_spec);
        return TensorSpec(
            input_tensor.logical_shape(),
            TensorLayout::fromPaddedShape(
                input_tensor.dtype(),
                PageConfig(Layout::TILE),
                mem_config,
                input_tensor.logical_shape(),
                input_tensor.padded_shape()));
    }

    return TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(Layout::TILE),
            operation_attributes.output_mem_config,
            input_tensor.logical_shape(),
            input_tensor.padded_shape()));
}

Tensor BcastDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    if (operation_attributes.in_place) {
        return tensor_args.input_a;
    }
    const spec_return_value_t spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(spec, tensor_args.input_a.device());
}

tt::stl::hash::hash_t BcastDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    log_trace(tt::LogOp, "BcastDeviceOperation::compute_program_hash is called");

    const bool bcast_scalar = (tensor_args.input_b.padded_shape()[-2] * tensor_args.input_b.padded_shape()[-1] == 1) &&
                              operation_attributes.dim == BcastOpDim::HW;

    auto program_factory = select_program_factory(operation_attributes, tensor_args);

    return operation::hash_operation<BcastDeviceOperation>(
        operation_attributes, tensor_args, bcast_scalar, program_factory.index());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor> BcastDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor0 = tensor_args.input_a;
    const Tensor& input_tensor1 = tensor_args.input_b;
    const Tensor& output_tensor = tensor_return_value;
    const bool output_is_dram = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_is_sharded = output_tensor.memory_config().is_sharded();
    const bool input_is_dram = input_tensor0.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool input_is_sharded = input_tensor0.memory_config().is_sharded();
    const bool is_local = input_is_sharded && !input_is_dram && output_is_sharded && !output_is_dram &&
                          (output_tensor.memory_config().shard_spec().value().grid ==
                           input_tensor0.memory_config().shard_spec().value().grid);
    const int ideal_dev_clock_cycles =
        common_tm_bw_model(input_tensor1, output_tensor, false, 0, false, false, is_local);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor0, input_tensor1}, tensor_return_value, ideal_dev_clock_cycles);
    return result;
}
}  // namespace ttnn::operations::data_movement::bcast

namespace ttnn::prim {
ttnn::operations::data_movement::bcast::BcastDeviceOperation::tensor_return_value_t bcast(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    ttnn::BcastOpMath bcast_op,
    ttnn::BcastOpDim bcast_dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool in_place,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::bcast::BcastDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .math_op = bcast_op, .dim = bcast_dim, .output_mem_config = output_mem_config, .in_place = in_place},
        OperationType::tensor_args_t{
            .input_a = input_tensor_a, .input_b = input_tensor_b, .preallocated_output = preallocated_output});
}
}  // namespace ttnn::prim

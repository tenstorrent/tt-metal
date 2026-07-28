// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/split/device/split_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

void SplitDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    // The program factory always chunks the last (width) dim; dim-2 splits are routed to the
    // slice fallback by the composite layer, so the device kernel only supports dim 3.
    TT_FATAL(args.dim == 3, "Split device kernel only supports the last dim (3)");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    // Sharded input/output are supported natively: the reader/writer kernels address by global
    // page_id and TensorAccessor resolves the physical NOC address for INTERLEAVED and
    // HEIGHT/WIDTH/BLOCK sharded buffers transparently.

    TT_FATAL(input_tensor.padded_shape().rank() == 4, "Tensor needs to be rank 4");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Tensor needs to be in TILE Layout");
    TT_FATAL(
        args.dim >= 0 && args.dim < static_cast<int>(input_tensor.padded_shape().rank()),
        "Dim being split must be from 0 to rank - 1");
    TT_FATAL(input_tensor.padded_shape()[0] == 1, "shape[0] must be 1 (batch 1 only)");
    TT_FATAL(
        input_tensor.padded_shape()[args.dim] % args.num_splits == 0,
        "Dim being split must be evenly divisible by number of splits");
    const uint32_t tile_size = (args.dim == 3) ? tt::constants::TILE_WIDTH : tt::constants::TILE_HEIGHT;
    TT_FATAL(
        (input_tensor.padded_shape()[args.dim] / tile_size) % args.num_splits == 0,
        "Tile count in split dim ({} tiles) must be divisible by num_splits ({})",
        input_tensor.padded_shape()[args.dim] / tile_size,
        args.num_splits);
}

SplitDeviceOperation::spec_return_value_t SplitDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto input_shape_array = input_tensor.padded_shape().to_array_4D();
    auto output_shape_array = input_shape_array;
    output_shape_array[args.dim] /= args.num_splits;
    tt::tt_metal::TensorSpec spec(
        Shape(output_shape_array),
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), args.output_mem_config));
    return std::vector<tt::tt_metal::TensorSpec>(args.num_splits, spec);
}

SplitDeviceOperation::tensor_return_value_t SplitDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    auto output_specs = compute_output_specs(args, tensor_args);

    std::vector<Tensor> output_tensors;
    output_tensors.reserve(args.num_splits);
    for (const auto& spec : output_specs) {
        output_tensors.push_back(create_device_tensor(spec, input_tensor.device()));
    }
    return output_tensors;
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SplitDeviceOperation::tensor_return_value_t>
SplitDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors) {
    const auto& input_tensor = tensor_args.input;
    std::vector<Tensor> input_tensors = {input_tensor};

    int ideal_dev_clock_cycles =
        operations::data_movement::common_tm_bw_model(input_tensor, output_tensors.at(0), false, 0, false, true);

    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
}

std::vector<ttnn::Tensor> split(
    const Tensor& input_tensor, int num_splits, int dim, const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::prim::SplitDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{num_splits, dim, output_mem_config},
        OperationType::tensor_args_t{input_tensor});
}

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include "pad_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/full/device/full_device_operation.hpp"
#include "ttnn/operations/creation.hpp"

#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_multi_core_v2_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_sharded_height_only_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_sharded_width_only_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_tile_multicore_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_tile_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::prim {
using ttnn::operations::data_movement::common_tm_bw_model;

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> PadDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& /*optional_input_tensors*/,
    std::vector<Tensor>& output_tensors) {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

PadDeviceOperation::program_factory_t PadDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        if (input_tensor.is_sharded()) {
            uint32_t input_tot_h = std::accumulate(
                input_tensor.logical_shape().view().begin(),
                input_tensor.logical_shape().view().end() - 1,
                1,
                std::multiplies<uint32_t>());
            uint32_t input_w = input_tensor.logical_shape()[3];

            uint32_t output_tot_h = std::accumulate(
                operation_attributes.output_logical_shape.view().begin(),
                operation_attributes.output_logical_shape.view().end() - 1,
                1,
                std::multiplies<uint32_t>());
            uint32_t output_w = operation_attributes.output_logical_shape[3];

            if (input_w != output_w and input_tot_h != output_tot_h) {
                TT_THROW(
                    "ttnn.pad: Unsupported sharded row-major padding configuration: pad_impl did not decompose padding "
                    "correctly.");
                return {};
            }
            if (input_w != output_w) {
                return PadRmShardedWidthOnlyProgramFactory{};
            }
            // height-only padding or no padding
            return PadRmShardedHeightOnlyProgramFactory{};
        }
        if (operation_attributes.use_multicore) {
            return PadRmReaderWriterMultiCoreV2ProgramFactory{};
        }
        return PadRmReaderWriterProgramFactory{};
    }
    if (input_tensor.layout() == Layout::TILE) {
        if (operation_attributes.use_multicore) {
            return PadTileMulticoreProgramFactory{};
        }
        return PadTileCoreProgramFactory{};
    }
    TT_THROW("Unsupported layout for pad");
    return {};
}

void PadDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(operation_attributes, tensor_args);
}

void PadDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const auto& input_tensor = tensor_args.input;
    auto logical_rank = input_tensor.logical_shape().rank();
    auto padded_rank = input_tensor.padded_shape().rank();
    TT_FATAL(logical_rank == padded_rank, "ttnn.pad: logical and padded shapes must have the same rank");
    TT_FATAL(input_tensor.logical_shape().rank() <= 4, "ttnn.pad: input tensor rank currently must be 4 or less");
    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operand to pad needs to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operand to pad needs to be allocated in a buffer on device!");
    TT_FATAL(
        input_tensor.layout() == tt::tt_metal::Layout::TILE || input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "Error");
    if (input_tensor.layout() == Layout::TILE) {
        TT_FATAL(
            (operation_attributes.input_tensor_start[0] == 0 && operation_attributes.input_tensor_start[1] == 0 &&
             operation_attributes.input_tensor_start[2] == 0 && operation_attributes.input_tensor_start[3] == 0),
            "On device padding only supports padding at end of dims");
    }
    TT_FATAL(
        input_tensor.padded_shape()[0] + operation_attributes.input_tensor_start[0] <=
            operation_attributes.output_padded_shape[0],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.padded_shape()[1] + operation_attributes.input_tensor_start[1] <=
            operation_attributes.output_padded_shape[1],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.padded_shape()[2] + operation_attributes.input_tensor_start[2] <=
            operation_attributes.output_padded_shape[2],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.padded_shape()[3] + operation_attributes.input_tensor_start[3] <=
            operation_attributes.output_padded_shape[3],
        "Output size cannot fit input with offset");

    if (input_tensor.layout() == Layout::TILE) {
        TT_FATAL(
            (operation_attributes.output_padded_shape[2] % TILE_HEIGHT == 0),
            "Can only pad tilized tensor with full tiles");
        TT_FATAL(
            (operation_attributes.output_padded_shape[3] % TILE_WIDTH == 0),
            "Can only pad tilized tensor with full tiles");
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
                input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                input_tensor.dtype() == DataType::UINT16,
            "Cannot pad tilized tensor with specified format");
    } else if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
                input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                input_tensor.dtype() == DataType::UINT16,
            "Cannot pad RM tensor with specified format");
    }

    if (input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "ttnn.pad: For sharded inputs, only height-sharding is supported.");
        TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "ttnn.pad: Only row-major sharded inputs are supported.");

        TT_FATAL(
            operation_attributes.output_mem_config.is_sharded(),
            "ttnn.pad: For sharded inputs, the output must be sharded.");
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
            "ttnn.pad: for sharded inputs, only height-sharding is supported for the output.");
    }
}

ttnn::TensorSpec PadDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    return TensorSpec(
        operation_attributes.output_logical_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            PageConfig(input_tensor.layout()),
            operation_attributes.output_mem_config,
            operation_attributes.output_logical_shape,
            operation_attributes.output_padded_shape));
}

Tensor PadDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }
    const auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input.device());
}

PadDeviceOperation::tensor_return_value_t pad(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool use_multicore,
    const std::optional<ttnn::Tensor>& preallocated_output) {
    using OperationType = PadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            output_logical_shape, output_padded_shape, input_tensor_start, pad_value, output_mem_config, use_multicore},
        OperationType::tensor_args_t{input, preallocated_output});
}
}  // namespace ttnn::prim

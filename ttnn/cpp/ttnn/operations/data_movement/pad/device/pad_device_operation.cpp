// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "pad_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/full/device/full_device_operation.hpp"
#include "ttnn/operations/creation.hpp"

#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_multi_core_default_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_sharded_height_only_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_sharded_width_only_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_tile_multicore_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_tile_program_factory.hpp"

using namespace tt::tt_metal;
namespace ttnn::prim {
using ttnn::operations::data_movement::common_tm_bw_model;

namespace {
bool can_use_sharded_optimized_factory(const PadParams& operation_attributes, const Tensor& input_tensor) {
    if (!input_tensor.shard_spec().has_value()) {
        return false;
    }
    if (input_tensor.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return false;
    }
    if (!operation_attributes.output_mem_config.is_sharded()) {
        return false;
    }
    if (operation_attributes.output_mem_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return false;
    }
    if (operation_attributes.sub_core_grids.has_value()) {
        return false;
    }
    if (operation_attributes.output_padded_shape[-1] == operation_attributes.output_mem_config.shard_spec()->shape[1]) {
        return false;
    }
    if (operation_attributes.output_mem_config.shard_spec().value().shape[0] <
        input_tensor.shard_spec().value().shape[0]) {
        // Note this case causes the sharded optimized PadRmShardedWidthOnlyProgramFactory{} to hang.
        return false;
    }
    return true;
}
}  // namespace

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
            if (can_use_sharded_optimized_factory(operation_attributes, input_tensor)) {
                uint32_t input_w = input_tensor.logical_shape()[3];
                uint32_t output_w = operation_attributes.output_logical_shape[3];
                uint32_t input_tot_h = std::accumulate(
                    input_tensor.logical_shape().view().begin(),
                    input_tensor.logical_shape().view().end() - 1,
                    1,
                    std::multiplies<uint32_t>());
                uint32_t output_tot_h = std::accumulate(
                    operation_attributes.output_logical_shape.view().begin(),
                    operation_attributes.output_logical_shape.view().end() - 1,
                    1,
                    std::multiplies<uint32_t>());

                if (input_w != output_w && input_tot_h == output_tot_h) {
                    return PadRmShardedWidthOnlyProgramFactory{};
                }
                if (input_w == output_w) {
                    return PadRmShardedHeightOnlyProgramFactory{};
                }
                // Combined width+height padding: fall through to the default factory
            }
            return PadRmReaderWriterMultiCoreDefaultProgramFactory{};
        }
        if (operation_attributes.use_multicore) {
            return PadRmReaderWriterMultiCoreDefaultProgramFactory{};
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
        input_tensor.logical_shape()[0] + operation_attributes.input_tensor_start[0] <=
            operation_attributes.output_padded_shape[0],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.logical_shape()[1] + operation_attributes.input_tensor_start[1] <=
            operation_attributes.output_padded_shape[1],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.logical_shape()[2] + operation_attributes.input_tensor_start[2] <=
            operation_attributes.output_padded_shape[2],
        "Output size cannot fit input with offset");
    TT_FATAL(
        input_tensor.logical_shape()[3] + operation_attributes.input_tensor_start[3] <=
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
                input_tensor.dtype() == DataType::UINT16 || input_tensor.dtype() == DataType::BFLOAT8_B,
            "Cannot pad tilized tensor with specified format");
    } else if (input_tensor.layout() == Layout::ROW_MAJOR) {
        TT_FATAL(
            input_tensor.dtype() == DataType::FLOAT32 || input_tensor.dtype() == DataType::BFLOAT16 ||
                input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                input_tensor.dtype() == DataType::UINT16,
            "Cannot pad RM tensor with specified format");
    }

    // Special conditions for sub_core_grids
    if (operation_attributes.sub_core_grids.has_value()) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "ttnn.pad: Input memory layout must be interleaved when sub_core_grids argument is provided");
        TT_FATAL(
            operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "ttnn.pad: Output memory layout must be interleaved when sub_core_grids argument is provided");
        TT_FATAL(
            operation_attributes.use_multicore,
            "ttnn.pad: sub_core_grids is only supported when use_multicore is true");
    }

    if (input_tensor.is_sharded()) {
        // TT_FATAL(input_tensor.layout() == Layout::ROW_MAJOR, "ttnn.pad: Only row-major sharded inputs are
        // supported.");

        uint32_t shard_width = input_tensor.shard_spec().has_value()
                                   ? input_tensor.shard_spec().value().shape[1]
                                   : input_tensor.nd_shard_spec().value().shard_shape[-1];
        const uint32_t page_size_bytes = input_tensor.buffer()->page_size();
        const uint32_t alignment_requirement = hal::get_l1_alignment();
        TT_FATAL(
            page_size_bytes == input_tensor.buffer()->aligned_page_size(),
            "Input row-major shard width {} gives page size {} bytes, which must be aligned to {} bytes",
            shard_width,
            page_size_bytes,
            alignment_requirement);
    }

    if (operation_attributes.output_mem_config.is_sharded()) {
        uint32_t output_shard_width =
            operation_attributes.output_mem_config.shard_spec().has_value()
                ? operation_attributes.output_mem_config.shard_spec().value().shape[1]
                : operation_attributes.output_mem_config.nd_shard_spec().value().shard_shape[-1];
        uint32_t output_page_size_bytes = output_shard_width * input_tensor.element_size();
        const uint32_t alignment_requirement = hal::get_l1_alignment();
        TT_FATAL(
            output_page_size_bytes == tt::align(output_page_size_bytes, alignment_requirement),
            "Output row-major shard width {} gives page size {} bytes, which must be aligned to {} bytes",
            output_shard_width,
            output_page_size_bytes,
            alignment_requirement);
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
    const std::optional<ttnn::Tensor>& preallocated_output,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    using OperationType = PadDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            output_logical_shape,
            output_padded_shape,
            input_tensor_start,
            pad_value,
            output_mem_config,
            use_multicore,
            sub_core_grids},
        OperationType::tensor_args_t{input, preallocated_output});
}
}  // namespace ttnn::prim

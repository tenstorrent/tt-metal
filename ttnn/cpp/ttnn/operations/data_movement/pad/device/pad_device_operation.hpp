// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/data_movement/pad/device/pad_device_operation_types.hpp"

#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_multi_core_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_multi_core_v2_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_reader_writer_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_sharded_height_only_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_rm_sharded_width_only_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_tile_multicore_program_factory.hpp"
#include "ttnn/operations/data_movement/pad/device/pad_tile_program_factory.hpp"

namespace ttnn::operations::data_movement::pad {
struct PadDeviceOperation {
    using operation_attributes_t = PadParams;
    using tensor_args_t = PadInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        program::PadRmReaderWriterMultiCoreProgramFactory,
        program::PadRmReaderWriterMultiCoreV2ProgramFactory,
        program::PadRmReaderWriterProgramFactory,
        program::PadRmShardedHeightOnlyProgramFactory,
        program::PadRmShardedWidthOnlyProgramFactory,
        program::PadTileMulticoreProgramFactory,
        program::PadTileCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors);

};
}  // namespace ttnn::operations::data_movement::pad

namespace ttnn::prim {
ttnn::operations::data_movement::pad::PadDeviceOperation::tensor_return_value_t pad(
    const Tensor& input,
    const ttnn::Shape& output_logical_shape,
    const ttnn::Shape& output_padded_shape,
    const ttnn::Shape& input_tensor_start,
    float pad_value,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool use_multicore,
    const std::optional<ttnn::Tensor>& preallocated_output = std::nullopt);
}  // namespace ttnn::prim

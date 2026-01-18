// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_single_core_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_interleaved_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_multi_core_sharded_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/tilize_with_val_padding_device_operation_types.hpp"

namespace ttnn::prim {

struct TilizeWithValPaddingDeviceOperation {
    using operation_attributes_t = TilizeWithValPaddingParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<
        ttnn::prim::TilizeWithValPaddingSingleCoreFactory,
        ttnn::prim::TilizeWithValPaddingMultiCoreBlockInterleavedFactory,
        ttnn::prim::TilizeWithValPaddingMultiCoreInterleavedFactory,
        ttnn::prim::TilizeWithValPaddingMultiCoreShardedFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const Tensor& input_tensor);
};

Tensor tilize_with_val_padding(
    const Tensor& input_tensor,
    const ttnn::Shape& output_padded_shape,
    const tt::tt_metal::PadValue& pad_value,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    bool use_multicore,
    bool enough_space_width,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids);
}  // namespace ttnn::prim

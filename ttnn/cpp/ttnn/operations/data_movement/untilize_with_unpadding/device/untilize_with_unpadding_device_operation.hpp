// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "factories/untilize_with_unpadding_single_core_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_sharded_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_col_interleaved_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include <variant>

namespace ttnn::prim {

struct UntilizeWithUnpaddingDeviceOperation {
    using operation_attributes_t = UntilizeWithUnpaddingParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<
        UntilizeWithUnpaddingSingleCoreProgramFactory,
        UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory,
        UntilizeWithUnpaddingMultiCoreShardedProgramFactory,
        UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory,
        UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const Tensor& input);

    static void validate_on_program_cache_hit(const operation_attributes_t& operation_attributes, const Tensor& input);

    static void validate_on_program_cache_miss(const operation_attributes_t& operation_attributes, const Tensor& input);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const Tensor& input);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const Tensor& input);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes, const Tensor& input, tensor_return_value_t& output_tensor);
};

Tensor untilize_with_unpadding(
    const Tensor& input_tensor,
    const ttnn::Shape& output_tensor_end,
    const std::optional<MemoryConfig>& output_mem_config,
    bool use_multicore,
    bool use_pack_untilize,
    bool fp32_dest_acc_en,
    bool enough_space_width,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim

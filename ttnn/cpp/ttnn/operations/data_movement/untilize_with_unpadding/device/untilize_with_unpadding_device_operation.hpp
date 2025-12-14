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

namespace ttnn::operations::data_movement {

struct UntilizeWithUnpaddingDeviceOperation {
    using operation_attributes_t =
        untilize_with_unpadding_types::operation_attributes_t;
    using tensor_args_t = untilize_with_unpadding_types::tensor_args_t;
    using spec_return_value_t = untilize_with_unpadding_types::spec_return_value_t;
    using tensor_return_value_t = untilize_with_unpadding_types::tensor_return_value_t;

    using program_factory_t = std::variant<
        detail::UntilizeWithUnpaddingSingleCoreProgramFactory,
        detail::UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory,
        detail::UntilizeWithUnpaddingMultiCoreShardedProgramFactory,
        detail::UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory,
        detail::UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const ttnn::Shape& output_tensor_end,
        const std::optional<MemoryConfig>& memory_config,
        const bool use_multicore,
        const bool use_pack_untilize,
        const bool fp32_dest_acc_en,
        const bool enough_space_width,
        const bool enough_space_height,
        const std::optional<CoreRangeSet>& sub_core_grids);
};

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
constexpr auto untilize_with_unpadding = ttnn::register_operation<
    "ttnn::prim::untilize_with_unpadding",
    ttnn::operations::data_movement::UntilizeWithUnpaddingDeviceOperation>();
}  // namespace ttnn::prim

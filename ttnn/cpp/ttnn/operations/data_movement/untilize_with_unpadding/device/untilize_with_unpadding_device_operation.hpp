// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"
#include "factories/untilize_with_unpadding_single_core_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_sharded_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"
#include "factories/untilize_with_unpadding_multi_core_nd_sharded_program_factory.hpp"
#include <variant>
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct UntilizeWithUnpaddingDeviceOperation {
    using operation_attributes_t = UntilizeWithUnpaddingParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<
        UntilizeWithUnpaddingSingleCoreProgramFactory,
        UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory,
        UntilizeWithUnpaddingMultiCoreShardedProgramFactory,
        UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory,
        UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const Tensor& input);

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
    bool fp32_dest_acc_en,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids);

// The data format to declare on a DFB of this op that holds elements of `dtype` on a device of `arch`.
// Quasar rejects UInt16 as a DFB format, so there a UInt16 element is declared as Int16, which has the
// same 2-byte tile layout. The compute kernels of this op unpack and pack each tile without computing on
// the values, so the 16-bit values pass through unchanged.
tt::DataFormat get_untilize_with_unpadding_dfb_data_format(tt::tt_metal::DataType dtype, tt::ARCH arch);

}  // namespace ttnn::prim

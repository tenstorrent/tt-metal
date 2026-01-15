// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "transpose_device_operation_types.hpp"
#include "transpose_cn_program_factory.hpp"
#include "transpose_hc_rm_program_factory.hpp"
#include "transpose_hc_sharded_program_factory.hpp"
#include "transpose_hc_tiled_interleaved_program_factory.hpp"
#include "transpose_hc_tiled_program_factory.hpp"
#include "transpose_wh_program_factory.hpp"
#include "transpose_wh_sharded_program_factory.hpp"
#include "transpose_wh_sharded_rm_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <variant>

namespace ttnn::operations::data_movement::transpose {

struct TransposeDeviceOperation {
    using operation_attributes_t = TransposeParams;
    using tensor_args_t = TransposeInputs;
    using spec_return_value_t = transpose::spec_return_value_t;
    using tensor_return_value_t = transpose::tensor_return_value_t;
    using program_factory_t = std::variant<
        program::TransposeWHProgramFactory,
        program::TransposeWHShardedProgramFactory,
        program::TransposeWHShardedRMProgramFactory,
        program::TransposeHCTiledInterleavedProgramFactory,
        program::TransposeHCTiledProgramFactory,
        program::TransposeHCRMProgramFactory,
        program::TransposeHCShardedProgramFactory,
        program::TransposeCNProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static TransposeOpParallelizationStrategy get_parallelization_strategy(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_hit(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, const Tensor& output);
};

}  // namespace ttnn::operations::data_movement::transpose

namespace ttnn::prim {
ttnn::Tensor transpose(
    const Tensor& input_tensor,
    ttnn::operations::data_movement::transpose::TransposeOpDim dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<float>& pad_value);
}  // namespace ttnn::prim

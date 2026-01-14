// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "concat_program_factory.hpp"
#include "concat_s2s_tiled_program_factory.hpp"
#include "concat_s2s_rm_program_factory.hpp"
#include "concat_s2s_multi_program_factory.hpp"
#include "concat_s2i_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include "concat_device_operation_types.hpp"

namespace ttnn::operations::data_movement::concat {

struct ConcatDeviceOperation {
    using operation_attributes_t = concat::operation_attributes_t;
    using tensor_args_t = concat::tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        program::ConcatProgramFactory,
        program::ConcatS2STiledProgramFactory,
        program::ConcatS2SRMProgramFactory,
        program::ConcatS2SMultiProgramFactory,
        program::ConcatS2IProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::operations::data_movement::concat

namespace ttnn::prim {
ttnn::operations::data_movement::concat::ConcatDeviceOperation::tensor_return_value_t concat(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

// Ref: https://pytorch.org/docs/stable/generated/torch.cat.html#torch.cat
// Notes: Non-empty tensors provided must have the same shape, except in the cat dimension.
Tensor concat_impl(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config);
}  // namespace ttnn::operations::data_movement

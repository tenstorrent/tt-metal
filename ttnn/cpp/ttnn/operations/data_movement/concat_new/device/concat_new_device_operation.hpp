// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "concat_new_program_factory.hpp"
#include "concat_new_s2s_tiled_program_factory.hpp"
#include "concat_new_s2s_rm_program_factory.hpp"
#include "concat_new_s2s_multi_program_factory.hpp"
#include "concat_new_s2i_program_factory.hpp"

#include "concat_new_device_operation_types.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct ConcatNewDeviceOperation {
    using operation_attributes_t = ConcatNewParams;
    using tensor_args_t = ConcatNewInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        ConcatNewProgramFactory,
        ConcatNewS2STiledProgramFactory,
        ConcatNewS2SRMProgramFactory,
        ConcatNewS2SMultiProgramFactory,
        ConcatNewS2IProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> create_op_performance_model(
        const std::vector<Tensor>& input_tensors,
        const std::vector<std::optional<const Tensor>>& optional_input_tensors,
        std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
ttnn::prim::ConcatNewDeviceOperation::tensor_return_value_t concat_new(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);
}  // namespace ttnn::prim

namespace ttnn::operations::data_movement {

Tensor concat_new_impl(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<ttnn::CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::operations::data_movement

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "argmax_device_operation_types.hpp"
#include "argmax_multi_core_program_factory.hpp"
#include "argmax_single_core_program_factory.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::prim {

struct ArgMaxDeviceOperation {
    using operation_attributes_t = ArgmaxParams;
    using tensor_args_t = ArgmaxInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ArgMaxSingleCoreProgramFactory, ArgMaxMultiCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t& args, const tensor_args_t&);
};

ttnn::Tensor argmax(
    const ttnn::Tensor& input,
    tt::tt_metal::DataType output_dtype,
    std::optional<int> dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool use_multicore,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    std::optional<ttnn::Tensor> optional_output_tensor = std::nullopt);

}  // namespace ttnn::prim

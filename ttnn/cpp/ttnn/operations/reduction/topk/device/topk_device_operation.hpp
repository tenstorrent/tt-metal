// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/topk/device/topk_device_operation_types.hpp"
#include "ttnn/operations/reduction/topk/device/topk_single_core_program_factory.hpp"
#include "ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp"

namespace ttnn::prim {

struct TopKDeviceOperation {
    using operation_attributes_t = TopkParams;
    using tensor_args_t = TopkInputs;
    using spec_return_value_t = std::tuple<TensorSpec, TensorSpec>;
    using tensor_return_value_t = std::tuple<Tensor, Tensor>;
    using program_factory_t = std::variant<TopKSingleCoreProgramFactory, TopKMultiCoreProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::prim

namespace ttnn::prim {
std::tuple<ttnn::Tensor, ttnn::Tensor> topk(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    const tt::tt_metal::MemoryConfig& memory_config,
    const tt::tt_metal::CoreRangeSet& sub_core_grids,
    const std::optional<Tensor>& indices_tensor = std::nullopt,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_output_tensors = std::nullopt);
}  // namespace ttnn::prim

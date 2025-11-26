// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "argmax_device_operation_types.hpp"
#include "argmax_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::argmax {

struct ArgMaxDeviceOperation {
    using operation_attributes_t = argmax::operation_attributes_t;
    using tensor_args_t = argmax::tensor_args_t;
    using spec_return_value_t = argmax::spec_return_value_t;
    using tensor_return_value_t = argmax::tensor_return_value_t;

    using program_factory_t = std::variant<
        program::ArgMaxSingleCoreRowMajorFactory,
        program::ArgMaxSingleCoreTileFactory,
        program::ArgMaxMultiCoreRowMajorFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        std::optional<int> dim,
        bool keepdim,
        const std::optional<CoreRangeSet>& sub_core_grids,
        bool use_multicore,
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_output_tensor);
};

}  // namespace ttnn::operations::reduction::argmax

namespace ttnn::prim {
constexpr auto argmax =
    ttnn::register_operation<"ttnn::prim::argmax", ttnn::operations::reduction::argmax::ArgMaxDeviceOperation>();
}  // namespace ttnn::prim

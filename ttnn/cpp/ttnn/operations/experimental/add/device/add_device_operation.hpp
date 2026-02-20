// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/experimental/add/device/add_device_operation_types.hpp"
#include "ttnn/operations/experimental/add/device/program_factory/element_wise_multi_core_add_program.hpp"
#include "ttnn/operations/experimental/add/device/program_factory/elt_nd_sharded_add_program.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <type_traits>
#include <variant>

namespace ttnn::experimental::prim {

template <typename T>
concept FloatOrTensorConcept = std::is_same_v<T, Tensor> || std::floating_point<T>;

struct AddDeviceOperation {
    using spec_return_value_t = TensorSpec;

    // Shared types with factory
    using operation_attributes_t = AddParams;
    using tensor_args_t = AddInputs;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<ElementWiseMultiCoreAddProgram, EltNDShardedAddProgram>;

    // DeviceOperationConcept methods
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attributes, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static bool skip_launch(const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& a_tensor,
        const Tensor& b_tensor,
        const std::optional<const DataType>& dtype,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> output_tensor);
};

}  // namespace ttnn::experimental::prim

// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_device_operation_types.hpp"
#include "isin_program_factory.hpp"

namespace ttnn::experimental::prim {

struct IsInDeviceOperation {
    using operation_attributes_t = IsinParams;
    using tensor_args_t = IsinInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<IsInProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor isin(
    const Tensor& elements,
    const Tensor& test_elements,
    uint32_t single_fetch_subchunk_size,
    bool assume_unique = false,
    bool invert = false,
    const std::optional<Tensor>& optional_out = std::nullopt);

}  // namespace ttnn::experimental::prim

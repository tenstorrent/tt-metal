// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_device_op_types.hpp"
#include "isin_program_factory.hpp"

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::isin {

struct IsInDeviceOperation {
    using operation_attributes_t = struct operation_attributes_t;
    using tensor_args_t = struct tensor_args_t;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<IsInProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;
    static invocation_result_t invoke(
        const Tensor& elements,
        const Tensor& test_elements,
        uint32_t single_fetch_subchunk_size,
        bool assume_unique,
        bool invert,
        const std::optional<Tensor>& optional_out);
};

}  // namespace ttnn::operations::experimental::isin

namespace ttnn::prim {

constexpr auto isin =
    ttnn::register_operation<"ttnn::prim::isin", ttnn::operations::experimental::isin::IsInDeviceOperation>();

}

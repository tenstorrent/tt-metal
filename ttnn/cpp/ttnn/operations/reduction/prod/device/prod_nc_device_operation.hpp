// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "prod_nc_device_operation_types.hpp"
#include "prod_nc_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::reduction::prod_nc {

struct ProdNcDeviceOperation {
    using operation_attributes_t = prod_nc::operation_attributes_t;
    using tensor_args_t = prod_nc::tensor_args_t;
    using spec_return_value_t = prod_nc::spec_return_value_t;
    using tensor_return_value_t = prod_nc::tensor_return_value_t;
    using program_factory_t = std::variant<program::ProdNcProgramFactory>;
    using shared_variables_t = program::ProdNcProgramFactory::shared_variables_t;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input, const Tensor& output, int64_t dim);
};

}  // namespace ttnn::operations::reduction::prod_nc

namespace ttnn::prim {
constexpr auto prod_nc =
    ttnn::register_operation<"ttnn::prim::prod_nc", ttnn::operations::reduction::prod_nc::ProdNcDeviceOperation>();
}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "prod_all_device_operation_types.hpp"
#include "prod_all_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct ProdAllDeviceOperation {
    using operation_attributes_t = ProdAllParams;
    using tensor_args_t = ProdAllInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ProdAllProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

ttnn::Tensor prod_all(const ttnn::Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "manual_seed_device_operation_types.hpp"
#include "manual_seed_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn::operations::reduction::manual_seed {

struct ManualSeedDeviceOperation {
    using operation_attributes_t = manual_seed::operation_attributes_t;
    using tensor_args_t = manual_seed::tensor_args_t;
    using spec_return_value_t = manual_seed::spec_return_value_t;
    using tensor_return_value_t = manual_seed::tensor_return_value_t;
    using program_factory_t = std::variant<program::ManualSeedProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        MeshDevice& device,
        std::variant<uint32_t, Tensor> seeds,
        std::optional<std::variant<uint32_t, Tensor>> user_ids);
};

}  // namespace ttnn::operations::reduction::manual_seed

namespace ttnn::prim {

constexpr auto manual_seed = ttnn::register_operation<
    "ttnn::prim::manual_seed",
    ttnn::operations::reduction::manual_seed::ManualSeedDeviceOperation>();

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "manual_seed_device_operation_types.hpp"
#include "manual_seed_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <functional>
#include <optional>

namespace ttnn::prim {

struct ManualSeedDeviceOperation {
    using operation_attributes_t = ManualSeedParams;
    using tensor_args_t = ManualSeedInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<
        ManualSeedSingleSeedToAllCoresProgramFactory,
        ManualSeedSingleSeedSingleCoreProgramFactory,
        ManualSeedSingleSeedSetCoresProgramFactory,
        ManualSeedSetSeedsSetCoresProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

ttnn::Tensor manual_seed(
    const std::variant<uint32_t, Tensor>& seeds,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<std::variant<uint32_t, Tensor>>& user_ids,
    const std::optional<CoreRangeSet>& sub_core_grids);

}  // namespace ttnn::prim

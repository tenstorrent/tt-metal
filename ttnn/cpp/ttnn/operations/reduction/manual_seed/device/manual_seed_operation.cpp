// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_operation.hpp"

#include "manual_seed/device/manual_seed_device_operation_types.hpp"

#include "tt_stl/assert.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/types.hpp"

#include <memory>

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::manual_seed {

ManualSeedDeviceOperation::program_factory_t ManualSeedDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return program::ManualSeedProgramFactory{};
}
void ManualSeedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        tensor_args.seeds.has_value() != operation_attributes.seeds.has_value(),
        "Either tensor_args.seeds or operation_attributes.seeds must be set, but not both.");

    if (tensor_args.seeds.has_value()) {
        TT_FATAL(
            !operation_attributes.user_ids.has_value(),
            "Seeds were provided as a tensor, so user_ids must not be provided as an scalar.");
        const auto& seeds_tensor = tensor_args.seeds.value();
        TT_FATAL(seeds_tensor.dtype() == DataType::UINT32, "Seeds tensor must be of type UINT32.");
        // TODO: More validations to be added when implementing device logic
    }
    if (tensor_args.user_ids.has_value()) {
        const auto& user_ids_tensor = tensor_args.user_ids.value();
        TT_FATAL(user_ids_tensor.dtype() == DataType::UINT32, "User IDs tensor must be of type UINT32.");
        TT_FATAL(
            !operation_attributes.user_ids.has_value(),
            "Either tensor_args.user_ids or operation_attributes.user_ids must be set, but not both.");
        // TODO: More validations to be added when implementing device logic
    }
    if (operation_attributes.seeds.has_value()) {
        TT_FATAL(
            !tensor_args.user_ids.has_value(),
            "Seeds were provided as a scalar, so user_ids must not be provided as an tensor.");
    }
}

void ManualSeedDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_hit(operation_attributes, tensor_args);
}

ManualSeedDeviceOperation::spec_return_value_t ManualSeedDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // NOTE: This OP does not return anything, but register_operation currently does not handle void return types.
    const TensorSpec tensor_spec(
        ttnn::Shape{1}, TensorLayout{DataType::UINT32, PageConfig{Layout::ROW_MAJOR}, MemoryConfig()});
    return tensor_spec;
}

ManualSeedDeviceOperation::tensor_return_value_t ManualSeedDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_specs = compute_output_specs(operation_attributes, tensor_args);
    // NOTE: This OP does not return anything, but register_operation currently does not handle void return types.
    return create_device_tensor(output_specs, operation_attributes.device);
}

std::tuple<ManualSeedDeviceOperation::operation_attributes_t, ManualSeedDeviceOperation::tensor_args_t>
ManualSeedDeviceOperation::invoke(
    MeshDevice& device, std::variant<uint32_t, Tensor> seeds, std::optional<std::variant<uint32_t, Tensor>> user_ids) {
    // Prepare operation attributes
    operation_attributes_t operation_attributes{};
    operation_attributes.device = std::addressof(device);
    if (std::holds_alternative<uint32_t>(seeds)) {
        operation_attributes.seeds = std::get<uint32_t>(seeds);
    }
    if (user_ids.has_value() && std::holds_alternative<uint32_t>(user_ids.value())) {
        operation_attributes.user_ids = std::get<uint32_t>(user_ids.value());
    }
    // Prepare tensor arguments
    tensor_args_t tensor_args{};

    // TODO: To be removed when API will be fixed with https://github.com/tenstorrent/tt-metal/pull/32260
    auto output_tensor = create_device_tensor(
        ttnn::TensorSpec(
            ttnn::Shape{1},
            tt::tt_metal::TensorLayout(DataType::UINT32, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), MemoryConfig())),
        std::addressof(device));
    tensor_args.output = output_tensor;

    if (std::holds_alternative<Tensor>(seeds)) {
        tensor_args.seeds = std::get<Tensor>(seeds);
    }
    if (user_ids.has_value() && std::holds_alternative<Tensor>(user_ids.value())) {
        tensor_args.user_ids = std::get<Tensor>(user_ids.value());
    }
    // Return prepared arguments
    return {operation_attributes, tensor_args};
}
}  // namespace ttnn::operations::reduction::manual_seed

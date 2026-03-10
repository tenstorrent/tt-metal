// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include "manual_seed/device/manual_seed_device_operation_types.hpp"

#include "tt_stl/assert.hpp"
#include "ttnn/tensor/layout/page_config.hpp"
#include "ttnn/tensor/types.hpp"

#include <memory>
#include <tt-logger/tt-logger.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

ManualSeedDeviceOperation::program_factory_t ManualSeedDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Case 1: seed=uint32_t, user_ids=None - set all cores to the same seed
    if (operation_attributes.seeds.has_value() && !operation_attributes.user_ids.has_value() &&
        !tensor_args.seeds.has_value() && !tensor_args.user_ids.has_value()) {
        return ManualSeedSingleSeedToAllCoresProgramFactory{};
    }
    // Case 2: seed=uint32_t, user_ids=uint32_t - set seed to one core based on user_id
    if (operation_attributes.seeds.has_value() && operation_attributes.user_ids.has_value() &&
        !tensor_args.seeds.has_value() && !tensor_args.user_ids.has_value()) {
        return ManualSeedSingleSeedSingleCoreProgramFactory{};
    }
    // Case 3: seed=uint32_t, user_ids=Tensor - set seeds to cores in user_ids tensor
    if (operation_attributes.seeds.has_value() && !operation_attributes.user_ids.has_value() &&
        !tensor_args.seeds.has_value() && tensor_args.user_ids.has_value()) {
        return ManualSeedSingleSeedSetCoresProgramFactory{};
    }
    // Case 4: seed=Tensor, user_ids=Tensor - set mapping seeds to cores based on tensors
    if (!operation_attributes.seeds.has_value() && !operation_attributes.user_ids.has_value() &&
        tensor_args.seeds.has_value() && tensor_args.user_ids.has_value()) {
        return ManualSeedSetSeedsSetCoresProgramFactory{};
    }
    log_warning(
        tt::LogMetal,
        "Logic error during selecting ManualSeed program factory, defaulting to "
        "ManualSeedSingleSeedToAllCoresProgramFactory");
    return ManualSeedSingleSeedToAllCoresProgramFactory{};
}
void ManualSeedDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Only one of seeds (tensor_args or operation_attributes) must be set
    TT_FATAL(
        tensor_args.seeds.has_value() != operation_attributes.seeds.has_value(),
        "Either tensor_args.seeds or operation_attributes.seeds must be set, but not both.");

    // Only one of user_ids (tensor_args or operation_attributes) can be set
    TT_FATAL(
        !(tensor_args.user_ids.has_value() && operation_attributes.user_ids.has_value()),
        "Either tensor_args.user_ids or operation_attributes.user_ids must be set, but not both.");

    // Seeds provided as tensor
    if (tensor_args.seeds.has_value()) {
        const auto& seeds_tensor = tensor_args.seeds.value();
        TT_FATAL(seeds_tensor.dtype() == DataType::UINT32, "Seeds tensor must be of type UINT32.");
        TT_FATAL(seeds_tensor.layout() == Layout::ROW_MAJOR, "Seeds tensor must have ROW_MAJOR layout.");
        // If user_ids are provided, they must also be a tensor
        if (tensor_args.user_ids.has_value()) {
            const auto& user_ids_tensor = tensor_args.user_ids.value();
            TT_FATAL(user_ids_tensor.dtype() == DataType::UINT32, "User IDs tensor must be of type UINT32.");
            TT_FATAL(user_ids_tensor.layout() == Layout::ROW_MAJOR, "User IDs tensor must have ROW_MAJOR layout.");
            TT_FATAL(
                seeds_tensor.logical_volume() == user_ids_tensor.logical_volume(),
                "Seeds tensor and User IDs tensor must have the same number of elements. Got seeds volume: {} and "
                "user_ids volume: {}",
                seeds_tensor.logical_volume(),
                user_ids_tensor.logical_volume());
            TT_FATAL(
                seeds_tensor.logical_shape() == user_ids_tensor.logical_shape(),
                "Seeds tensor and User IDs tensor must have the same shape.");
            TT_FATAL(
                seeds_tensor.logical_shape().rank() == 1, "Seeds tensor and User IDs tensor must be 1-dimensional.");
        }
        // If operation_attributes.user_ids is set, error
        TT_FATAL(
            !operation_attributes.user_ids.has_value(),
            "Seeds were provided as a tensor, so user_ids must not be provided as a scalar.");
    }
    // Seeds provided as scalar (operation_attributes)
    if (operation_attributes.seeds.has_value()) {
        if (operation_attributes.user_ids.has_value()) {
            TT_FATAL(
                operation_attributes.user_ids.value() >= 0 && operation_attributes.user_ids.value() <= 31,
                "User IDs scalar must be in the range [0, 31].");
        }
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

ttnn::Tensor manual_seed(
    const std::variant<uint32_t, Tensor>& seeds,
    std::optional<std::reference_wrapper<MeshDevice>> device,
    const std::optional<std::variant<uint32_t, Tensor>>& user_ids,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    if (std::holds_alternative<uint32_t>(seeds)) {
        TT_FATAL(device.has_value(), "Device must be provided when seeds is a uint32_t value.");
    }

    using OperationType = ManualSeedDeviceOperation;
    OperationType::operation_attributes_t operation_attributes{};
    if (device.has_value()) {
        operation_attributes.device = std::addressof(device.value().get());
    } else {
        const auto& seeds_tensor = std::get<Tensor>(seeds);
        operation_attributes.device = seeds_tensor.device();
    }

    if (std::holds_alternative<uint32_t>(seeds)) {
        operation_attributes.seeds = std::get<uint32_t>(seeds);
    }
    if (user_ids.has_value() && std::holds_alternative<uint32_t>(user_ids.value())) {
        operation_attributes.user_ids = std::get<uint32_t>(user_ids.value());
    }
    operation_attributes.sub_core_grids = sub_core_grids;

    OperationType::tensor_args_t tensor_args{};
    if (std::holds_alternative<Tensor>(seeds)) {
        tensor_args.seeds = std::get<Tensor>(seeds);
    }
    if (user_ids.has_value() && std::holds_alternative<Tensor>(user_ids.value())) {
        tensor_args.user_ids = std::get<Tensor>(user_ids.value());
    }

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim

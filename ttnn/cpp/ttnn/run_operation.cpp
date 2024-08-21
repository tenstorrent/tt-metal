// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_utils.hpp>

#include "ttnn/operations/experimental/auto_format/auto_format.hpp"
#include "ttnn/operation.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/third_party/tracy/public/tracy/Tracy.hpp"
#include "tt_metal/tools/profiler/op_profiler.hpp"
#include "tt_metal/tt_stl/reflection.hpp"
#include "ttnn/config.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"

using ttnn::operations::experimental::auto_format::AutoFormat;

namespace tt::tt_metal::operation {

namespace detail {

inline bool any_tensor_on_multi_device(const Tensors& tensors) {
    return std::any_of(tensors.begin(), tensors.end(), [](const Tensor& tensor) {
        return tensor.storage_type() == StorageType::MULTI_DEVICE;
    });
}

Device* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors) {
    for (auto& input_tensor : input_tensors) {
        if (std::holds_alternative<DeviceStorage>(input_tensor.tensor_attributes->storage)) {
            return input_tensor.workers.at(0);
        }
    }
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and
            std::holds_alternative<DeviceStorage>(optional_input_tensor.value().tensor_attributes->storage)) {
            return optional_input_tensor.value().workers.at(0);
        }
    }
    auto device = AutoFormat::GetDefaultDevice();
    TT_ASSERT(device != nullptr, "Requires setting default device if no inputs to operation are on device");
    return device;
}

template <class OutputTensors>
void override_addresses(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program& program,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OutputTensors& output_tensors) {
    std::vector<tt::tt_metal::Buffer*> input_buffers;
    for (auto& tensor : input_tensors) {
        input_buffers.push_back(tensor.buffer());
    }
    for (auto& tensor : optional_input_tensors) {
        auto buffer = tensor.has_value() ? tensor.value().buffer() : nullptr;
        input_buffers.push_back(buffer);
    }

    std::vector<tt::tt_metal::Buffer*> output_buffers;
    for (auto& tensor : output_tensors) {
        if constexpr (std::is_same_v<OptionalTensors, OutputTensors>) {
            auto buffer = tensor.has_value() ? tensor.value().buffer() : nullptr;
            output_buffers.push_back(buffer);
        } else {
            output_buffers.push_back(tensor.buffer());
        }
    }

    override_addresses_callback(program, input_buffers, output_buffers);
}

template void override_addresses<Tensors>(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program& program,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const Tensors& output_tensors);

template void override_addresses<OptionalTensors>(
    const OverrideAddressesCallback& override_addresses_callback,
    const Program& program,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& output_tensors);

}  // namespace detail

template<typename OutputTensors>
struct OldInfraDeviceOperation {

    using operation_attributes_t = operation::DeviceOperation<OutputTensors>;

    struct tensor_args_t {
        const operation::Tensors input_tensors;
        const operation::OptionalConstTensors optional_input_tensors;
        const operation::OptionalTensors optional_output_tensors;
    };

    using shape_return_value_t = std::vector<tt::tt_metal::Shape>;

    using tensor_return_value_t = OutputTensors;

    struct ProgramFactory {
        struct shared_variables_t {
            std::optional<operation::OverrideAddressesCallback>  override_addresses_callback;
            std::optional<operation::OverrideRuntimeArgumentsCallback<OutputTensors>> override_runtime_arguments_callback;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto program_with_callbacks = operation_attributes.create_program(
                tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_return_value);
            return cached_program_t{
                std::move(program_with_callbacks.program),
                shared_variables_t{
                    program_with_callbacks.override_addresses_callback,
                    program_with_callbacks.override_runtime_arguments_callback}
            };
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            auto& override_addresses_callback = cached_program.shared_variables.override_addresses_callback;
            auto& override_runtime_arguments_callback = cached_program.shared_variables.override_runtime_arguments_callback;
            auto& program = cached_program.program;

            if (override_addresses_callback.has_value()) {
                // Deprecated
                detail::override_addresses(
                    override_addresses_callback.value(),
                    program,
                    tensor_args.input_tensors,
                    tensor_args.optional_input_tensors,
                    tensor_return_value);
            }

            if (override_runtime_arguments_callback.has_value()) {
                operation_attributes.override_runtime_arguments(
                    override_runtime_arguments_callback.value(),
                    program,
                    tensor_args.input_tensors,
                    tensor_args.optional_input_tensors,
                    tensor_return_value);
            }
        }
    };

    using program_factory_t = std::variant<ProgramFactory>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return ProgramFactory{};
    }

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        attributes.validate(tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_args.optional_output_tensors);
    }

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        validate_on_program_cache_miss(attributes, tensor_args);
    }

    // Compute the output shapes based on the operation attributes and tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return attributes.compute_output_shapes(tensor_args.input_tensors);
    }

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return attributes.create_output_tensors(tensor_args.input_tensors, tensor_args.optional_output_tensors);
    }

     static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
        return attributes.compute_program_hash(tensor_args.input_tensors, tensor_args.optional_input_tensors);
    }

    static auto create_op_performance_model(
        const operation_attributes_t& attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value) {
        return attributes.create_op_performance_model(tensor_args.input_tensors, tensor_args.optional_input_tensors, tensor_return_value);
    }

    static std::string get_type_name(const operation_attributes_t& attributes) {
        return attributes.get_type_name();
    }

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        operation_attributes_t&& operation_attributes,
        const operation::Tensors& input_tensors,
        const operation::OptionalConstTensors& optional_input_tensors,
        const operation::OptionalTensors& optional_output_tensors
    ) {
        return std::make_tuple(
            std::move(operation_attributes),
            tensor_args_t{input_tensors, optional_input_tensors, optional_output_tensors}
        );
    }
};


} // namespace tt::tt_metal::operation

namespace ttnn::prim {
constexpr auto old_infra_device_operation = ttnn::register_operation<
    "ttnn::prim::old_infra_device_operation",
    tt::tt_metal::operation::OldInfraDeviceOperation<tt::tt_metal::operation::Tensors>>();
constexpr auto old_infra_device_operation_with_optional_output_tensors = ttnn::register_operation<
    "ttnn::prim::old_infra_device_operation_with_optional_output_tensors",
    tt::tt_metal::operation::OldInfraDeviceOperation<tt::tt_metal::operation::OptionalTensors>>();
}  // namespace ttnn::prim

namespace tt::tt_metal::operation {

template <class OutputTensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id) {

    if constexpr (std::is_same_v<OutputTensors, Tensors>) {
        return ttnn::prim::old_infra_device_operation(cq_id, std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    } else {
        return ttnn::prim::old_infra_device_operation_with_optional_output_tensors(cq_id, std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors);
    }
}

template Tensors run(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

template OptionalTensors run(
    DeviceOperation<OptionalTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

template <class OutputTensors>
OutputTensors run_without_autoformat(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id) {
    ZoneScoped;
    Device* device = detail::get_device(input_tensors, optional_input_tensors);
    Tensors input_tensors_on_dev;
    input_tensors_on_dev.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        if (input_tensor.storage_type() != StorageType::DEVICE) {
            input_tensors_on_dev.push_back(AutoFormat::move_tensor_to_device(input_tensor, device));
        } else {
            input_tensors_on_dev.push_back(input_tensor);
        }
    }
    OptionalConstTensors optional_input_tensors_on_dev;
    optional_input_tensors_on_dev.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value() and optional_input_tensor.value().storage_type() != StorageType::DEVICE) {
            optional_input_tensors_on_dev.push_back(
                AutoFormat::move_tensor_to_device(optional_input_tensor.value(), device));
        } else {
            optional_input_tensors_on_dev.push_back(optional_input_tensor);
        }
    }
    return run<OutputTensors>(std::move(operation), input_tensors_on_dev, optional_input_tensors_on_dev, optional_output_tensors, cq_id);
}

template Tensors run_without_autoformat<Tensors>(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

template OptionalTensors run_without_autoformat<OptionalTensors>(
    DeviceOperation<OptionalTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id);

// To be deprecated/removed in favor of new implementation where ops specifically request how to format inputs/outputss
Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors,
    const OptionalTensors& optional_output_tensors,
    const float pad_value,
    const bool pad_c,
    uint8_t cq_id) {
    ZoneScoped;
    Device* device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_shapes = operation.compute_output_shapes(input_tensors);

    Tensors formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (auto& input_tensor : input_tensors) {
        auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape(), pad_c);
        auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
        if (pad_input) {
            formatted_input_tensors.push_back(
                AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
        } else {
            formatted_input_tensors.push_back(input_tensor);
        }
    }

    OptionalConstTensors formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (auto& optional_input_tensor : optional_input_tensors) {
        if (optional_input_tensor.has_value()) {
            auto& input_tensor = optional_input_tensor.value();
            auto padded_input_shape = AutoFormat::pad_to_tile_shape(input_tensor.get_legacy_shape(), pad_c);
            auto pad_input = not AutoFormat::check_input_tensor_format(input_tensor, padded_input_shape);
            if (pad_input) {
                formatted_optional_input_tensors.push_back(
                    AutoFormat::format_input_tensor(input_tensor, device, padded_input_shape, pad_value, Layout::TILE));
            } else {
                formatted_optional_input_tensors.push_back(input_tensor);
            }
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensor);
        }
    }

    auto output_tensors = run<Tensors>(std::move(operation), formatted_input_tensors, formatted_optional_input_tensors, optional_output_tensors, cq_id);

    TT_ASSERT(output_tensors.size() == output_shapes.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] = AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, Layout::TILE);
    }
    return output_tensors;
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors,
    const std::vector<std::optional<FormatParams>>& optional_input_formatting,
    const OptionalTensors& optional_output_tensors,
    uint8_t cq_id) {
    ZoneScoped;
    Device* device = detail::get_device(input_tensors, optional_input_tensors);
    auto output_shapes = operation.compute_output_shapes(input_tensors);

    TT_ASSERT(input_tensors.size() == input_formatting.size());
    TT_ASSERT(optional_input_tensors.size() == optional_input_formatting.size());

    Tensors formatted_input_tensors;
    formatted_input_tensors.reserve(input_tensors.size());
    for (uint32_t i = 0; i < input_tensors.size(); ++i) {
        formatted_input_tensors.push_back(AutoFormat::format_input_tensor(
            input_tensors[i],
            device,
            input_formatting[i].pad_shape,
            input_formatting[i].pad_value,
            input_formatting[i].target_layout));
    }

    OptionalConstTensors formatted_optional_input_tensors;
    formatted_optional_input_tensors.reserve(optional_input_tensors.size());
    for (uint32_t i = 0; i < optional_input_tensors.size(); ++i) {
        if (optional_input_tensors[i].has_value()) {
            auto& input_tensor = optional_input_tensors[i].value();
            TT_ASSERT(optional_input_formatting[i].has_value());
            auto& input_formatting = optional_input_formatting[i].value();
            formatted_optional_input_tensors.push_back(AutoFormat::format_input_tensor(
                input_tensor,
                device,
                input_formatting.pad_shape,
                input_formatting.pad_value,
                input_formatting.target_layout));
        } else {
            formatted_optional_input_tensors.push_back(optional_input_tensors[i]);
        }
    }

    auto output_tensors = run<Tensors>(std::move(operation), formatted_input_tensors, formatted_optional_input_tensors, optional_output_tensors, cq_id);

    TT_ASSERT(output_tensors.size() == output_shapes.size());
    TT_ASSERT(output_tensors.size() == output_layouts.size());

    formatted_input_tensors.clear();
    formatted_optional_input_tensors.clear();

    for (auto i = 0; i < output_tensors.size(); ++i) {
        output_tensors[i] =
            AutoFormat::format_output_tensor(output_tensors[i], output_shapes[i], device, output_layouts[i]);
    }

    return output_tensors;
}

void launch_with_autoformat(
    std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>&& op_func,
    const Tensors input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors optional_input_tensors,
    const OptionalTensors optional_output_tensors) {
    // Mark each output tensor as having dynamic storage (can be on host or device, depending
    // on autoformat behaviour). Multi device tensors do not support dynamic storage.
    for (auto& output_tensor : output_tensors) {
        output_tensor.tensor_attributes->dynamic_storage = (output_tensor.workers.size() <= 1);
    }
    launch_op(std::move(op_func), input_tensors, output_tensors, optional_input_tensors, optional_output_tensors);
}


void validate_workers_and_storage(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs,
    const std::vector<Device*>& workers) {
    bool single_device_storage = false;
    bool multi_device_storage = false;
    // Verify that storage types are consistent - cannot mix single and multi-device storage. For multi-device tensors,
    // ensure that workers are specified, since they cannot be inferred. This means that
    // launch_op/launch_with_autoformat cannot be called with MultiDeviceHostStorage.
    for (const auto& input : inputs) {
        if (std::holds_alternative<DeviceStorage>(input.tensor_attributes->storage) or
            std::holds_alternative<OwnedStorage>(input.tensor_attributes->storage)) {
            single_device_storage |= true;
        } else if (
            std::holds_alternative<MultiDeviceStorage>(input.tensor_attributes->storage) or
            std::holds_alternative<MultiDeviceHostStorage>(input.tensor_attributes->storage)) {
            multi_device_storage |= true;
        }
    }

    for (auto& input : optional_inputs) {
        if (input.has_value()) {
            if (std::holds_alternative<DeviceStorage>(input.value().tensor_attributes->storage) or
                std::holds_alternative<OwnedStorage>(input.value().tensor_attributes->storage)) {
                single_device_storage |= true;
            } else if (
                std::holds_alternative<MultiDeviceStorage>(input.value().tensor_attributes->storage) or
                std::holds_alternative<MultiDeviceHostStorage>(input.value().tensor_attributes->storage)) {
                multi_device_storage |= true;
            }
        }
    }

    TT_FATAL(
        not(single_device_storage and multi_device_storage),
        "Cannot mix single and multi-device tensors when calling launch op!");
    if (multi_device_storage) {
        TT_FATAL(
            workers.size(),
            "Workers must be specified when calling launch_op with with multi-device tensors. Workers cannot be "
            "inferred in this case.");
    }
}

std::vector<Device*> get_workers_for_op_output(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs,
    bool enable_autoformat_device) {
    std::vector<Device*> workers_for_op = {};
    // Infer output workers from inputs. For multi-device tensors the number
    // of workers used for the op (and assigned to the ouput) is the minimum
    // number of workers across all inputs. Additionally, in this case, at least
    // 1 worker must be specified across all inputs, i.e. host inputs are not allowed.
    size_t min_workers_size = std::numeric_limits<uint32_t>::max();
    for (auto& input : inputs) {
        auto workers = input.get_workers();
        min_workers_size = std::min(min_workers_size, workers.size());
        if (workers.size() == min_workers_size) {
            workers_for_op = workers;
        }
    }

    if (not workers_for_op.size()) {
        for (auto& input : optional_inputs) {
            if (input.has_value()) {
                auto workers = input.value().get_workers();
                min_workers_size = std::min(min_workers_size, workers.size());
                if (workers.size() == min_workers_size) {
                    workers_for_op = workers;
                }
            }
        }
    }
    if (enable_autoformat_device) {
        validate_workers_and_storage(inputs, optional_inputs, workers_for_op);
        // Workers not specified - inputs are on host and not multi-device.
        // Use the default device from autoformat.
        if (not workers_for_op.size()) {
            TT_FATAL(
                AutoFormat::GetDefaultDevice(),
                "Default device must be specified using AutoFormat::SetDefaultDevice, if workers are not specified for "
                "inputs to op.");
            workers_for_op = {AutoFormat::GetDefaultDevice()};
        }
    }
    return workers_for_op;
}
}  // namespace tt::tt_metal::operation

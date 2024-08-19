// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>

#include "ttnn/deprecated/tt_dnn/op_library/auto_format.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operation_history.hpp"
#include "tt_stl/concepts.hpp"
#include "tt_stl/type_name.hpp"

namespace tt::tt_metal {

namespace operation {

template <typename ConcreteOperation>
auto generic_create_output_tensors(
    const ConcreteOperation& operation,
    const Tensors& input_tensors,
    const std::optional<DataType> output_dtype,
    const Layout output_layout,
    const std::optional<MemoryConfig>& output_mem_config) -> ProgramOutputTensors<ConcreteOperation> {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_shapes = operation.compute_output_shapes(input_tensors);

    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    OutputTensors output_tensors;
    output_tensors.reserve(output_shapes.size());
    for (const auto& output_shape : output_shapes) {
        output_tensors.emplace_back(create_device_tensor(
            output_shape,
            output_dtype.value_or(input_tensors.at(0).get_dtype()),
            output_layout,
            input_tensor.device(),
            output_mem_config.value_or(input_tensors.at(0).memory_config())));
    }
    return output_tensors;
}

#ifdef DEBUG
namespace detail {

template <typename OperationType>
std::string operation_type_to_string() {
 if constexpr (std::is_same_v<OperationType, DeviceOperation<Tensors>>) {
        return "device<Tensors>";
    }else if constexpr (std::is_same_v<OperationType, DeviceOperation<OptionalTensors>>) {
        return "device<OptionalTensors>";
    } else if constexpr (std::is_same_v<OperationType, ExternalOperation>) {
        return "external";
    } else {
        static_assert(tt::stl::concepts::always_false_v<OperationType>, "OperationType is not supported!");
    }
}

template <typename OperationType>
static void append_operation_to_operation_history(
    const std::size_t ttnn_operation_id,
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors) {

    std::vector<operation_history::TensorRecord> input_tensor_records;
    input_tensor_records.reserve(input_tensors.size() + optional_input_tensors.size());

    for (const auto& tensor : input_tensors) {
        input_tensor_records.emplace_back(operation_history::create_tensor_record(tensor));
    }
    for (const auto& tensor : optional_input_tensors) {
        if (tensor.has_value()) {
            input_tensor_records.emplace_back(operation_history::create_tensor_record(tensor.value()));
        }
    }

    std::optional<bool> program_cache_hit = std::nullopt;
    std::optional<tt::stl::hash::hash_t> program_hash = std::nullopt;
    if constexpr (std::is_same_v<OperationType, DeviceOperation<typename OperationType::OutputTensors>>) {
        auto& program_cache = input_tensors[0].device()->program_cache;
        if (program_cache.is_enabled()) {
            program_hash = operation.compute_program_hash(input_tensors, optional_input_tensors);
            auto program_cache_hit = program_cache.contains(program_hash.value());
        }
    }

    operation_history::append(operation_history::OperationRecord{
        ttnn_operation_id,
        std::string(tt::stl::short_type_name<OperationType>),
        operation.get_type_name(),
        operation.attributes(),
        input_tensor_records,
        program_cache_hit,
        program_hash,
    });
}

}  // namespace detail

template<typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) {
    tt::log_debug(
        tt::LogOp,
        "Launching Operation: \"{}\" ({})",
        operation.get_type_name(),
        detail::operation_type_to_string<OperationType>());

    auto attributes = operation.attributes();
    if (not attributes.empty()) {
        tt::log_debug(tt::LogOp, "Attributes:");
        for (auto&& [name, value] : attributes) {
            tt::log_debug(tt::LogOp, "\t{} = {}", name, value);
        }
    }

    tt::log_debug(tt::LogOp, "Input Tensors:");
    for (auto index = 0; index < input_tensors.size(); index++) {
        const auto& tensor = input_tensors[index];
        tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
    }

    if (not optional_input_tensors.empty()) {
        tt::log_debug(tt::LogOp, "Optional Input Tensors:");
        for (auto index = 0; index < optional_input_tensors.size(); index++) {
            const auto& tensor = optional_input_tensors[index];
            tt::log_debug(tt::LogOp, "\t{}: {}", index, tensor);
        }
    }

    tt::log_debug(tt::LogOp, "");

    if (operation_history::enabled()) {
        detail::append_operation_to_operation_history(
            ttnn::OPERATION_ID, operation, input_tensors, optional_input_tensors);
    }
}
#else

template <typename OperationType>
inline void log_operation(
    const OperationType& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {}) {}
#endif

template<class OutputTensors=Tensors>
OutputTensors run(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0);

template<typename ConcreteOperation>
inline auto run(
    ConcreteOperation&& concrete_op,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors={},
    const OptionalTensors& optional_output_tensors={},
    uint8_t cq_id = 0
) -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        auto operation = DeviceOperation(concrete_op);
        return run<OutputTensors>(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

template<class OutputTensors=Tensors>
OutputTensors run_without_autoformat(
    DeviceOperation<OutputTensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0
);
template <typename ConcreteOperation>
inline auto run_without_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {},
    uint8_t cq_id = 0)
    -> ProgramOutputTensors<ConcreteOperation>{
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_without_autoformat<OutputTensors>(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const OptionalConstTensors& optional_input_tensors = {},
    const OptionalTensors& optional_output_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false,
    uint8_t cq_id = 0
);

template<typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<Tensor>>& optional_output_tensors = {},
    const float pad_value = 0,
    const bool pad_c = false,
    uint8_t cq_id = 0
)-> Tensors {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<Tensors>(concrete_op);
    return run_with_autoformat(std::move(operation), input_tensors, optional_input_tensors, optional_output_tensors, pad_value, pad_c, cq_id);
}

Tensors run_with_autoformat(
    DeviceOperation<Tensors>&& operation,
    const Tensors& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const OptionalConstTensors& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0
);
template<typename ConcreteOperation>
inline auto run_with_autoformat(
    ConcreteOperation&& concrete_op,
    const std::vector<Tensor>& input_tensors,
    const std::vector<FormatParams>& input_formatting,
    const std::vector<Layout>& output_layouts,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors = {},
    const std::vector<std::optional<FormatParams>>& optional_input_formatting = {},
    const OptionalTensors& optional_output_tensors = {},
    uint8_t cq_id = 0
)-> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_with_autoformat(std::move(operation), input_tensors, input_formatting, output_layouts, optional_input_tensors, optional_input_formatting, optional_output_tensors, cq_id);
}

template<class Callable, class OutputType=Tensors>
void launch_op(
    Callable&& op_func,
    const Tensors input_tensors,
    OutputType& output_tensors,
    const OptionalConstTensors optional_input_tensors = {},
    const OptionalTensors optional_output_tensors = {},
    bool enable_autoformat_device = true
);

void launch_with_autoformat(
    std::function<Tensors(const Tensors&, const OptionalConstTensors&, const OptionalTensors&)>&& op_func,
    const Tensors input_tensors,
    Tensors& output_tensors,
    const OptionalConstTensors optional_input_tensors = {},
    const OptionalTensors optional_output_tensors = {}
);

std::vector<Device*> get_workers_for_op_output(
    const std::vector<Tensor>& inputs,
    const std::vector<std::optional<const Tensor>>& optional_inputs = {},
    bool enable_autoformat_device = true);

namespace detail{
    Device* get_device(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors = {});
}

} //namespace operation

} //namespace tt::tt_metal

#include"run_operation_inl.hpp"

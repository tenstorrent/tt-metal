// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <ttnn/tensor/tensor.hpp>

#include "ttnn/experimental/tt_dnn/op_library/auto_format.hpp"
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

namespace run_operation_state {
namespace detail {

struct RunOperationState {

    RunOperationState() {}

    void push_composite_parent_name(const char* parent_name) {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        this->composite_parent_names.push_back(parent_name);
    }

    void pop_composite_parent_name() {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        this->composite_parent_names.pop_back();
    }

    bool is_composite_operation() const {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        return not composite_parent_names.empty();
    }

    const auto& get_composite_parent_names() const {
        std::scoped_lock<std::mutex> lock(parent_name_mutex);
        return this->composite_parent_names;
    }

  private:
    mutable std::mutex parent_name_mutex;
    std::vector<const char*> composite_parent_names{};
};

inline RunOperationState OPERATION_STATE{};

}  // namespace detail

inline void push_composite_parent_name(const char* parent_name) {
    detail::OPERATION_STATE.push_composite_parent_name(parent_name);
}

inline void pop_composite_parent_name() {
    detail::OPERATION_STATE.pop_composite_parent_name();
}

inline bool is_composite_operation() {
    return detail::OPERATION_STATE.is_composite_operation();
}

inline const auto& get_composite_parent_names() {
    return detail::OPERATION_STATE.get_composite_parent_names();
}

}  // namespace run_operation_state


namespace detail {

template<typename ReturnType, typename... Args>
struct CompositeOperation {

    const char* name;
    std::function<ReturnType(Args...)> function;

    constexpr ReturnType operator()(Args... args) const {
        run_operation_state::push_composite_parent_name(this->name);
        ReturnType output = this->function(args...);
        run_operation_state::pop_composite_parent_name();
        return output;
    }
};

}  // namespace detail

template<typename ReturnType, typename... Args>
constexpr auto decorate_as_composite(const char* name, std::function<ReturnType(Args...)>&& function) {
  return detail::CompositeOperation<ReturnType, Args...>{.name=name, .function=function};
}

template<typename FunctionType>
constexpr auto decorate_as_composite(const char* name, FunctionType function) {
  return decorate_as_composite(name, std::function(function));
}

#ifdef DEBUG
namespace detail {

template <typename OperationType>
std::string operation_type_to_string() {
    if constexpr (std::is_same_v<OperationType, HostOperation<Tensors>>) {
        return "host<Tensors>";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation<Tensors>>) {
        return "device<Tensors>";
    } else if constexpr (std::is_same_v<OperationType, HostOperation<OptionalTensors>>) {
        return "host<OptionalTensors>";
    } else if constexpr (std::is_same_v<OperationType, DeviceOperation<OptionalTensors>>) {
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
        run_operation_state::get_composite_parent_names(),
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

    if (run_operation_state::is_composite_operation()) {
        tt::log_debug(tt::LogOp, "Composite Parents: {}", run_operation_state::get_composite_parent_names());
    }

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
    const HostOperation<OutputTensors>& operation,
    const Tensors& input_tensors
);

template<class OutputTensors=Tensors>
OutputTensors run(
    const DeviceOperation<OutputTensors>& operation,
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
    if constexpr (detail::is_host_operation<ConcreteOperation>()) {
        TT_ASSERT(optional_input_tensors.empty());
        const auto operation = HostOperation(concrete_op);
        return run<OutputTensors>(operation, input_tensors);
    } else if constexpr (detail::is_device_operation<ConcreteOperation>()) {
        const auto operation = DeviceOperation(concrete_op);
        return run<OutputTensors>(operation, input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
    } else {
        static_assert(tt::stl::concepts::always_false_v<ConcreteOperation>, "Unsupported Operation");
    }
}

template<class OutputTensors=Tensors>
OutputTensors run_without_autoformat(
    const DeviceOperation<OutputTensors>& operation,
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
    const auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_without_autoformat<OutputTensors>(operation, input_tensors, optional_input_tensors, optional_output_tensors, cq_id);
}

Tensors run_with_autoformat(
    const DeviceOperation<Tensors>& operation,
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
    const auto operation = DeviceOperation<Tensors>(concrete_op);
    return run_with_autoformat(operation, input_tensors, optional_input_tensors, optional_output_tensors, pad_value, pad_c, cq_id);
}

Tensors run_with_autoformat(
    const DeviceOperation<Tensors>& operation,
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
    const auto operation = DeviceOperation<OutputTensors>(concrete_op);
    return run_with_autoformat(operation, input_tensors, input_formatting, output_layouts, optional_input_tensors, optional_input_formatting, optional_output_tensors, cq_id);
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

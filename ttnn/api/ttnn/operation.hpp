// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <experimental/type_traits>
#include <ttnn/tensor/tensor.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/config.hpp"
#include "ttnn/distributed/types.hpp"

#include <tracy/Tracy.hpp>

namespace tt {

namespace tt_metal {
namespace operation {

using Hash = tt::stl::hash::hash_t;

template <typename OperationType, typename... Types>
static Hash hash_operation(const Types&... objects) {
    return stl::hash::hash_objects_with_default_seed(tt::stl::hash::type_hash<OperationType>, objects...);
}

using Tensors = std::vector<Tensor>;
using OptionalTensors = std::vector<std::optional<Tensor>>;
using OptionalConstTensors = std::vector<std::optional<const Tensor>>;

template <typename OutputTensors = Tensors>
using OverrideRuntimeArgumentsCallback = std::function<void(
    const void* operation, Program&, const Tensors&, const OptionalConstTensors&, const OutputTensors&)>;

template <typename OutputTensors = Tensors>
struct CacheableProgram {
    Program program;
    std::optional<OverrideRuntimeArgumentsCallback<OutputTensors>> override_runtime_arguments_callback = std::nullopt;
};

template <typename OutputTensors = Tensors>
using OverrideRuntimeArgumentsWorkloadCallback = std::function<void(
    const void* operation,
    distributed::MeshWorkload&,
    const Tensors&,
    const OptionalConstTensors&,
    const OutputTensors&)>;

template <typename OutputTensors = Tensors>
struct CacheableMeshWorkload {
    distributed::MeshWorkload workload;

    // Either one of these callbacks can be set, but not both.
    // TODO: #19569 - `per_program_callbacks` is used to assist old infra migration, which relied on per-program
    // callbacks. This needs to be removed
    std::optional<OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>> workload_callback = std::nullopt;
    std::unordered_map<ttnn::MeshCoordinateRange, OverrideRuntimeArgumentsCallback<OutputTensors>>
        per_program_callbacks;
};

template <typename... Args>
struct last_type;

template <typename T>
struct last_type<T> {
    using type = T;
};

template <typename First, typename... Rest>
struct last_type<First, Rest...> : last_type<Rest...> {};

// An alias template to map to a function
template <class TReturn, class... TArgs>
using fn = TReturn(TArgs...) const;

template <class FnPtr>
struct function_traits;

// "T::" means member of
// * means pointer
// fn<TReturn, TArgs...> represents the function member type
// conceptually it is TReturn (T::*)(TArgs...) but fn<TReturn, TArgs...>
// allows us to then get to TReturn
template <class T, class TReturn, class... TArgs>
struct function_traits<fn<TReturn, TArgs...> T::*> {
    using return_t = TReturn;
    using last_arg_t = typename last_type<TArgs...>::type;
};

// Just grab the last arg from the function_traits
template <class FnPtr>
using last_arg_of_function_t = typename function_traits<FnPtr>::last_arg_t;

template <class FnPtr>
using return_arg_of_function_t = typename function_traits<FnPtr>::return_t;

template <typename, typename = std::void_t<>>
struct has_create_program : std::false_type {};

template <typename ConcreteOperation>
struct has_create_program<ConcreteOperation, std::void_t<decltype(&ConcreteOperation::create_program)>>
    : std::true_type {};

template <typename ConcreteOperation, bool HasCreateProgram = has_create_program<ConcreteOperation>::value>
struct program_output_helper;

// If we have create_program, then we need to use the last argument for the OutputTensors
template <typename ConcreteOperation>
struct program_output_helper<ConcreteOperation, true> {
    using type = std::remove_const_t<std::remove_reference_t<
        last_arg_of_function_t<decltype(&std::remove_reference<ConcreteOperation>::type::create_program)>>>;
};

// If create_program does not exist on the ConcreteOperation this specialization will fallback to Tensors
template <typename ProgramType>
struct program_output_helper<ProgramType, false> {
    using type = Tensors;
};

template <typename ProgramType>
using ProgramOutputTensors = typename program_output_helper<ProgramType>::type;

template <class OutputTensorsT = Tensors>
struct OpPerformanceModelGeneral {
    using OutputTensors = OutputTensorsT;
    int ideal_compute_cycles = 1;
    int ideal_compute_ns = 1;
    int ideal_bandwidth_ns = 1;
    int ideal_ns = 1;
    std::vector<int> inputs_bytes = {};
    std::vector<int> outputs_bytes = {};

    OpPerformanceModelGeneral(Tensors input_tensors, OutputTensors output_tensors, int ideal_compute_cycles) {
        const auto& t = input_tensors.at(0);
        const auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : ARCH::WORMHOLE_B0;

        this->ideal_compute_cycles = ideal_compute_cycles;

        float clock_rate_ghz = (arch == ARCH::WORMHOLE_B0) ? 1.0 : 1.2;
        this->ideal_compute_ns = std::ceil(ideal_compute_cycles / clock_rate_ghz);

        // GS L1 Bisection bandwidth
        // 655 B/cycle = sqrt(108) * 32 B/cycle * 2
        // 655 * 1.2Ghz = 786 GB/s
        // GS DRAM bandwidth
        // 96 GB/s = 12 GB/s * 8 channels

        // WH L1 Bisection bandwidth
        // 512 B/cycle = sqrt(64) * 32 B/cycle * 2
        // 512 * 1ghz clk
        // WH DRAM bandwidth
        // 258 GB/s = 21.5 GB/s * 6 channels * 2 banks

        float peak_dram_bw = (arch == ARCH::WORMHOLE_B0) ? 6 * 2 * 21.5 : 96.0;

        float noc_l1_bisection_bw = (arch == ARCH::WORMHOLE_B0) ? 512.0 : 786.0;

        auto tensor_ns = [peak_dram_bw, noc_l1_bisection_bw](const Tensor& t) {
            int size_bytes = t.physical_volume() * t.element_size();
            if (t.memory_config().is_dram()) {
                return size_bytes / peak_dram_bw / 1024 / 1024 / 1024 * 1000 * 1000 * 1000;
            } else if (t.memory_config().is_l1()) {
                return 1.0f;  // TODO: figure out better modelling scheme for L1->L1 Transfers
                // return size_bytes / noc_l1_bisection_bw / 1024 / 1024 / 1024 * 1000 * 1000 * 1000;
            }
            return 0.0f;
        };

        for (const auto& t : input_tensors) {
            this->inputs_bytes.push_back(t.physical_volume() * t.element_size());
            if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                this->ideal_bandwidth_ns = tensor_ns(t);
            }
        }
        if constexpr (std::is_same_v<OutputTensors, Tensors>) {
            for (const auto& t : output_tensors) {
                this->outputs_bytes.push_back(t.physical_volume() * t.element_size());
                if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                    this->ideal_bandwidth_ns = tensor_ns(t);
                }
            }
        } else if constexpr (std::is_same_v<OutputTensors, Tensor>) {
            this->outputs_bytes.push_back(output_tensors.physical_volume() * output_tensors.element_size());
        } else {
            for (const auto& ot : output_tensors) {
                if (!ot.has_value()) {
                    continue;
                }
                auto& t = ot.value();
                this->outputs_bytes.push_back(t.physical_volume() * t.element_size());
                if (tensor_ns(t) > this->ideal_bandwidth_ns) {
                    this->ideal_bandwidth_ns = tensor_ns(t);
                }
            }
        }

        this->ideal_ns = std::max(this->ideal_compute_ns, this->ideal_bandwidth_ns);
    }
    OpPerformanceModelGeneral() = default;
    ~OpPerformanceModelGeneral() = default;

    int get_compute_ns() const { return this->ideal_compute_ns; }
    int get_ideal_ns() const { return this->ideal_ns; }
    int get_bandwidth_ns() const { return this->ideal_bandwidth_ns; }
    std::vector<float> get_input_bws() const {
        std::vector<float> input_bws(inputs_bytes.size());
        TT_ASSERT(this->ideal_ns > 0);
        std::transform(inputs_bytes.cbegin(), inputs_bytes.cend(), input_bws.begin(), [this](float c) {
            return (float)c / this->ideal_ns;
        });
        return input_bws;
    }
    std::vector<float> get_output_bws() const {
        std::vector<float> output_bws(outputs_bytes.size());
        TT_ASSERT(this->ideal_ns > 0);
        std::transform(outputs_bytes.cbegin(), outputs_bytes.cend(), output_bws.begin(), [this](float c) {
            return (float)c / this->ideal_ns;
        });
        return output_bws;
    }

    static int fidelity_multiplier(MathFidelity f) {
        if (MathFidelity::LoFi == f) {
            return 1;
        } else if (MathFidelity::HiFi2 == f) {
            return 2;
        } else if (MathFidelity::HiFi3 == f) {
            return 3;
        } else if (MathFidelity::HiFi4 == f) {
            return 4;
        }

        return 0;
    }
};

using OpPerformanceModel = OpPerformanceModelGeneral<>;

struct ProfilerInfo {
    std::optional<std::string> preferred_name;
    std::optional<std::string> parallelization_strategy;
};

inline MemoryConfig DEFAULT_OUTPUT_MEMORY_CONFIG;

inline void set_default_operation_output_memory_config(const MemoryConfig& memory_config) {
    DEFAULT_OUTPUT_MEMORY_CONFIG = memory_config;
}

namespace detail {

// TODO: move 'NotImplemented' to a library file
class NotImplemented : public std::logic_error {
public:
    NotImplemented(const std::string& message) : std::logic_error(message) {};
};

template <class T, class... Args>
using has_get_type_name_t = decltype(std::declval<T>().get_type_name(std::declval<Args>()...));

template <class T>
constexpr bool implements_get_type_name() {
    return std::experimental::is_detected_v<has_get_type_name_t, T>;
}
template <class T, class... Args>
using has_validate_t = decltype(std::declval<T>().validate(std::declval<Args>()...));

template <class T>
constexpr bool implements_validate() {
    return std::experimental::is_detected_v<has_validate_t, T, const Tensors&>;
}

template <class T>
constexpr bool implements_validate_with_optional_input_tensors() {
    return std::experimental::
        is_detected_v<has_validate_t, T, const Tensors&, const std::vector<std::optional<const Tensor>>&>;
}

template <class T, class... Args>
using has_validate_with_output_tensors_t =
    decltype(std::declval<T>().validate_with_output_tensors(std::declval<Args>()...));

template <class T>
constexpr bool implements_validate_with_output_tensors() {
    return std::experimental::is_detected_v<
        has_validate_with_output_tensors_t,
        T,
        const Tensors&,           // input_tensors
        const OptionalTensors&>;  // optional output_tensors
}

template <class T>
constexpr bool implements_validate_with_output_tensors_and_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_validate_with_output_tensors_t,
        T,
        const Tensors&,                                   // input_tensors
        const std::vector<std::optional<const Tensor>>&,  // optional input_tensors
        const OptionalTensors&>;                          // optional output_tensors
}

template <class T, class... Args>
using has_compute_output_specs_t = decltype(std::declval<T>().compute_output_specs(std::declval<Args>()...));

template <class T>
constexpr bool implements_compute_output_specs_with_optional_output_tensors() {
    return std::experimental::is_detected_v<has_compute_output_specs_t, T, const Tensors&, const OptionalTensors&>;
}

template <class T>
constexpr bool implements_compute_output_specs() {
    return std::experimental::is_detected_v<has_compute_output_specs_t, T, const Tensors&> ||
           implements_compute_output_specs_with_optional_output_tensors<T>();
}

template <class T, class... Args>
using has_create_output_tensors_t = decltype(std::declval<T>().create_output_tensors(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_output_tensors() {
    return std::experimental::is_detected_v<has_create_output_tensors_t, T, const Tensors&>;
}

template <class T>
constexpr bool implements_create_output_tensors_with_optional_output_tensors() {
    return std::experimental::is_detected_v<has_create_output_tensors_t, T, const Tensors&, const OptionalTensors&>;
}

template <class T, class... Args>
using has_create_program_t = decltype(std::declval<T>().create_program(std::declval<Args>()...));

template <class T, class... Args>
using has_create_mesh_workload_t = decltype(std::declval<T>().create_mesh_workload(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_program() {
    return std::experimental::is_detected_v<has_create_program_t, T, const Tensors&, Tensors&> or
           std::experimental::is_detected_v<has_create_program_t, T, const Tensors&, OptionalTensors&>;
}

template <class T>
constexpr bool implements_create_program_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
               has_create_program_t,
               T,
               const Tensors&,
               const std::vector<std::optional<const Tensor>>&,
               Tensors&> or
           std::experimental::is_detected_v<
               has_create_program_t,
               T,
               const Tensors&,
               const std::vector<std::optional<const Tensor>>&,
               OptionalTensors&>;
}

template <class T>
constexpr bool implements_create_mesh_workload() {
    return std::experimental::is_detected_v<
               has_create_mesh_workload_t,
               T,
               const ttnn::MeshCoordinateRangeSet&,
               const Tensors&,
               Tensors&> or
           std::experimental::is_detected_v<
               has_create_mesh_workload_t,
               T,
               const ttnn::MeshCoordinateRangeSet&,
               const Tensors&,
               OptionalTensors&>;
}

template <class T>
constexpr bool implements_create_mesh_workload_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
               has_create_mesh_workload_t,
               T,
               const ttnn::MeshCoordinateRangeSet&,
               const Tensors&,
               const std::vector<std::optional<const Tensor>>&,
               Tensors&> or
           std::experimental::is_detected_v<
               has_create_mesh_workload_t,
               T,
               const ttnn::MeshCoordinateRangeSet&,
               const Tensors&,
               const std::vector<std::optional<const Tensor>>&,
               OptionalTensors&>;
}

template <class T, class... Args>
using has_create_op_performance_model_t =
    decltype(std::declval<T>().create_op_performance_model(std::declval<Args>()...));

template <class T>
constexpr bool implements_create_op_performance_model() {
    return std::experimental::is_detected_v<
        has_create_op_performance_model_t,
        T,
        const Tensors&,
        const std::vector<std::optional<const Tensor>>&,
        Tensors&>;
}

template <class T, class... Args>
using has_compute_program_hash_t = decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template <class T>
constexpr bool implements_compute_program_hash() {
    return std::experimental::is_detected_v<has_compute_program_hash_t, T, const Tensors&>;
}

template <class T, class... Args>
using has_compute_program_hash_with_optional_input_tensors_t =
    decltype(std::declval<T>().compute_program_hash(std::declval<Args>()...));

template <class T>
constexpr bool implements_compute_program_hash_with_optional_input_tensors() {
    return std::experimental::is_detected_v<
        has_compute_program_hash_with_optional_input_tensors_t,
        T,
        const Tensors&,
        const std::vector<std::optional<const Tensor>>&>;
}

template <class T>
constexpr bool is_device_operation() {
    return implements_create_program<T>() || implements_create_mesh_workload<T>() ||
           implements_create_program_with_optional_input_tensors<T>() ||
           implements_create_mesh_workload_with_optional_input_tensors<T>();
}

template <class T, class... Args>
using has_get_parallelization_strategy_t =
    decltype(std::declval<T>().get_parallelization_strategy(std::declval<Args>()...));

template <class T>
constexpr bool implements_get_parallelization_strategy() {
    return std::experimental::is_detected_v<has_get_parallelization_strategy_t, T, const Tensors&>;
}

}  // namespace detail

template <typename ConcreteOperation>
auto default_create_output_tensors(
    const ConcreteOperation& operation, const Tensors& input_tensors, const OptionalTensors& optional_output_tensors)
    -> ProgramOutputTensors<ConcreteOperation> {
    using OutputTensors = ProgramOutputTensors<ConcreteOperation>;
    OutputTensors output_tensors;

    if (!optional_output_tensors.empty() and optional_output_tensors[0].has_value()) {
        output_tensors.reserve(optional_output_tensors.size());
        for (const auto& optional_output_tensor : optional_output_tensors) {
            TT_FATAL(
                optional_output_tensor.has_value(),
                "If using optional output tensors, all output tensors must have a value");
            output_tensors.emplace_back(optional_output_tensor.value());
        }
        return output_tensors;
    }
    const auto& device = input_tensors.at(0).device();
    const auto& output_specs = operation.compute_output_specs(input_tensors);
    output_tensors.reserve(output_specs.size());
    for (const auto& output_spec : output_specs) {
        output_tensors.emplace_back(create_device_tensor(output_spec, device));
    }
    return output_tensors;
}

template <class OutputTensorsT = Tensors>
class DeviceOperation final {
public:
    using storage_t = std::array<std::byte, 1152>;
    using OutputTensors = OutputTensorsT;
    using ComputedSpecs = std::vector<ttnn::TensorSpec>;

    std::string get_type_name() const { return this->get_type_name_impl_(this->type_erased_storage); }

    void validate(
        const Tensors& input_tensors,
        const OptionalConstTensors& optional_input_tensors,
        const OptionalTensors& optional_output_tensors) const {
        return this->validate_impl_(
            this->type_erased_storage, input_tensors, optional_input_tensors, optional_output_tensors);
    }

    ComputedSpecs compute_output_specs(const Tensors& input_tensors, const OptionalTensors& output_tensors) const {
        return this->compute_output_specs_impl_(this->type_erased_storage, input_tensors, output_tensors);
    }

    OutputTensors create_output_tensors(const Tensors& input_tensors, const OptionalTensors& output_tensors) const {
        return this->create_output_tensors_impl_(this->type_erased_storage, input_tensors, output_tensors);
    }

    CacheableProgram<OutputTensors> create_program(
        const Tensors& input_tensors,
        const OptionalConstTensors& optional_input_tensors,
        OutputTensors& output_tensors) const {
        return this->create_program_impl_(
            this->type_erased_storage, input_tensors, optional_input_tensors, output_tensors);
    }

    CacheableMeshWorkload<OutputTensors> create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensors& input_tensors,
        const OptionalConstTensors& optional_input_tensors,
        OutputTensors& output_tensors) const {
        return this->create_mesh_workload_impl_(
            this->type_erased_storage, tensor_coords, input_tensors, optional_input_tensors, output_tensors);
    }

    bool has_create_workload_method() const { return this->has_create_workload_method_impl_(); }

    OpPerformanceModelGeneral<OutputTensors> create_op_performance_model(
        const Tensors& input_tensors,
        const OptionalConstTensors& optional_input_tensors,
        OutputTensors& output_tensors) const {
        return this->create_op_performance_model_impl_(
            this->type_erased_storage, input_tensors, optional_input_tensors, output_tensors);
    }

    void override_runtime_arguments(
        OverrideRuntimeArgumentsCallback<OutputTensors>& override_runtime_arguments_callback,
        Program& program,
        const Tensors& input_tensors,
        const OptionalConstTensors& optional_input_tensors,
        OutputTensors& output_tensors) const {
        return this->override_runtime_arguments_program_impl_(
            this->type_erased_storage,
            override_runtime_arguments_callback,
            program,
            input_tensors,
            optional_input_tensors,
            output_tensors);
    }

    void override_runtime_arguments(
        OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>& override_runtime_arguments_callback,
        distributed::MeshWorkload& workload,
        const Tensors& input_tensors,
        const OptionalConstTensors& optional_input_tensors,
        OutputTensors& output_tensors) const {
        return this->override_runtime_arguments_workload_impl_(
            this->type_erased_storage,
            override_runtime_arguments_callback,
            workload,
            input_tensors,
            optional_input_tensors,
            output_tensors);
    }

    bool uses_custom_program_hash() const { return this->uses_custom_program_hash_impl_(); }

    Hash compute_program_hash(const Tensors& input_tensors, const OptionalConstTensors& optional_input_tensors) const {
        ZoneScoped;
        return this->compute_program_hash_impl_(this->type_erased_storage, input_tensors, optional_input_tensors);
    }

    ProfilerInfo create_profiler_info(const Tensors& input_tensors) const {
        return this->create_profiler_info_impl_(this->type_erased_storage, input_tensors);
    }

    tt::stl::reflection::Attributes attributes() const { return this->attributes_impl_(this->type_erased_storage); }

    template <typename T>
        requires(not std::same_as<std::decay_t<T>, DeviceOperation<OutputTensorsT>>)
    explicit DeviceOperation(T&& operation) :

        pointer{new(&type_erased_storage) std::decay_t<T>{std::forward<T>(operation)}},

        delete_storage{[](storage_t& self) {
            using Type = std::decay_t<T>;
            reinterpret_cast<Type*>(&self)->~Type();
        }},
        copy_storage{[](storage_t& self, const void* other) -> void* {
            using Type = std::decay_t<T>;
            if constexpr (std::is_copy_constructible_v<Type>) {
                return new (&self) Type{*reinterpret_cast<const Type*>(other)};
            } else {
                static_assert(tt::stl::concepts::always_false_v<Type>);
            }
        }},
        move_storage{[](storage_t& self, void* other) -> void* {
            using Type = std::decay_t<T>;
            if constexpr (std::is_move_constructible_v<Type>) {
                return new (&self) Type{*reinterpret_cast<Type*>(other)};
            } else {
                static_assert(tt::stl::concepts::always_false_v<Type>);
            }
        }},

        // Initialize methods
        get_type_name_impl_{[](const storage_t& storage) -> std::string {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
            if constexpr (detail::implements_get_type_name<T>()) {
                return operation.get_type_name();
            } else {
                return std::string(tt::stl::get_type_name<T>());
            }
        }},
        validate_impl_{
            [](const storage_t& storage,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors,
               const OptionalTensors& optional_output_tensors) -> void {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (
                    (detail::implements_validate<T>() or
                     detail::implements_validate_with_optional_input_tensors<T>()) and
                    (detail::implements_validate_with_output_tensors<T>() or
                     detail::implements_validate_with_output_tensors_and_optional_input_tensors<T>())) {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>,
                        "You cannot implement both validate and validate_with_output_tensors");
                } else if constexpr (
                    detail::implements_validate<T>() and
                    not(detail::implements_create_program<T>() || detail::implements_create_mesh_workload<T>())) {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>,
                        "Operation doesn't implement both the validate and the correct create_program or "
                        "create_mesh_workload methods");
                } else if constexpr (
                    detail::implements_validate_with_optional_input_tensors<T>() and
                    not(detail::implements_create_program_with_optional_input_tensors<T>() ||
                        detail::implements_create_mesh_workload_with_optional_input_tensors<T>())) {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>,
                        "Operation doesn't implement both the validate and the correct create_program or "
                        "create_mesh_workload methods with the "
                        "optional input tensors");
                }

                if constexpr (detail::implements_validate<T>()) {
                    TT_FATAL(optional_input_tensors.empty(), "Optional input tensors not allowed");
                    operation.validate(input_tensors);
                } else if constexpr (detail::implements_validate_with_optional_input_tensors<T>()) {
                    TT_FATAL(not optional_input_tensors.empty(), "Optional input tensors are expected");
                    operation.validate(input_tensors, optional_input_tensors);
                } else if constexpr (detail::implements_validate_with_output_tensors<T>()) {
                    TT_FATAL(optional_input_tensors.empty(), "Optional input tensors not allowed");
                    // TT_FATAL(not optional_output_tensors.empty(), "Error");
                    operation.validate_with_output_tensors(input_tensors, optional_output_tensors);
                } else if constexpr (detail::implements_validate_with_output_tensors_and_optional_input_tensors<T>()) {
                    TT_FATAL(not optional_input_tensors.empty(), "Optional input tensors are expected");
                    TT_FATAL(not optional_output_tensors.empty(), "Optional output tensors are expected");
                    operation.validate_with_output_tensors(
                        input_tensors, optional_input_tensors, optional_output_tensors);
                } else {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>,
                        "Operation must implement either validate or validate_with_output_tensors");
                }
            }},
        compute_output_specs_impl_{
            [](const storage_t& storage,
               const Tensors& input_tensors,
               const OptionalTensors& output_tensors) -> ComputedSpecs {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_compute_output_specs_with_optional_output_tensors<T>()) {
                    return operation.compute_output_specs(input_tensors, output_tensors);
                } else if constexpr (detail::implements_compute_output_specs<T>()) {
                    return operation.compute_output_specs(input_tensors);
                } else {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>, "Operation must implement compute_output_specs");
                }
            }},
        create_output_tensors_impl_{
            [](const storage_t& storage,
               const Tensors& input_tensors,
               const OptionalTensors& output_tensors) -> OutputTensors {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_output_tensors_with_optional_output_tensors<T>()) {
                    static_assert(
                        detail::implements_compute_output_specs<T>(),
                        "Operation must implement compute_output_specs if it implements create_output_tensors");
                    return operation.create_output_tensors(input_tensors, output_tensors);
                } else if constexpr (detail::implements_create_output_tensors<T>()) {
                    static_assert(
                        detail::implements_compute_output_specs<T>(),
                        "Operation must implement compute_output_specs if it implements create_output_tensors");
                    return operation.create_output_tensors(input_tensors);
                } else if constexpr (detail::implements_compute_output_specs<T>()) {
                    return default_create_output_tensors(operation, input_tensors, output_tensors);
                } else {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>,
                        "Operation must implement either create_output_tensors or compute_output_specs");
                }
            }},
        create_program_impl_{
            [](const storage_t& storage,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors,
               OutputTensors& output_tensors) -> CacheableProgram<OutputTensors> {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_program<T>()) {
                    TT_FATAL(
                        optional_input_tensors.empty(),
                        "Optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return operation.create_program(input_tensors, output_tensors);
                } else if constexpr (detail::implements_create_program_with_optional_input_tensors<T>()) {
                    TT_FATAL(
                        not optional_input_tensors.empty(),
                        "Non-optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return operation.create_program(input_tensors, optional_input_tensors, output_tensors);
                } else {
                    TT_THROW("Operation doesn't implement create_program");
                }
            }},
        create_mesh_workload_impl_{
            [](const storage_t& storage,
               const ttnn::MeshCoordinateRangeSet& tensor_coords,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors,
               OutputTensors& output_tensors) -> CacheableMeshWorkload<OutputTensors> {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_mesh_workload<T>()) {
                    TT_FATAL(
                        optional_input_tensors.empty(),
                        "Optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return operation.create_mesh_workload(tensor_coords, input_tensors, output_tensors);
                } else if constexpr (detail::implements_create_mesh_workload_with_optional_input_tensors<T>()) {
                    TT_FATAL(
                        not optional_input_tensors.empty(),
                        "Non-optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return operation.create_mesh_workload(
                        tensor_coords, input_tensors, optional_input_tensors, output_tensors);
                } else {
                    TT_THROW("Operation doesn't implement create_mesh_workload");
                }
            }},
        create_op_performance_model_impl_{
            [](const storage_t& storage,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors,
               OutputTensors& output_tensors) -> OpPerformanceModelGeneral<OutputTensors> {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                if constexpr (detail::implements_create_op_performance_model<T>()) {
                    return operation.create_op_performance_model(input_tensors, optional_input_tensors, output_tensors);
                } else {
                    return OpPerformanceModelGeneral<OutputTensors>(
                        input_tensors, output_tensors, 1);  // TODO: account for optional_input_tensors
                }
            }},
        override_runtime_arguments_program_impl_{
            [](const storage_t& storage,
               OverrideRuntimeArgumentsCallback<OutputTensors>& callback,
               Program& program,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors,
               OutputTensors& output_tensors) -> void {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                callback(&operation, program, input_tensors, optional_input_tensors, output_tensors);
            }},
        override_runtime_arguments_workload_impl_{
            [](const storage_t& storage,
               OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>& callback,
               distributed::MeshWorkload& workload,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors,
               OutputTensors& output_tensors) -> void {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
                callback(&operation, workload, input_tensors, optional_input_tensors, output_tensors);
            }},
        compute_program_hash_impl_{
            [](const storage_t& storage,
               const Tensors& input_tensors,
               const OptionalConstTensors& optional_input_tensors) -> Hash {
                const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);

                if constexpr (detail::implements_compute_program_hash<T>()) {
                    static_assert(
                        detail::implements_create_program<T>() || detail::implements_create_mesh_workload<T>());
                    TT_FATAL(
                        optional_input_tensors.empty(),
                        "Optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return operation.compute_program_hash(input_tensors);
                } else if constexpr (detail::implements_compute_program_hash_with_optional_input_tensors<T>()) {
                    static_assert(
                        detail::implements_create_program_with_optional_input_tensors<T>() ||
                        detail::implements_create_mesh_workload_with_optional_input_tensors<T>());
                    TT_FATAL(
                        not optional_input_tensors.empty(),
                        "Non-optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return operation.compute_program_hash(input_tensors, optional_input_tensors);
                } else if constexpr (
                    detail::implements_create_program<T>() || detail::implements_create_mesh_workload<T>()) {
                    TT_FATAL(
                        optional_input_tensors.empty(),
                        "Optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return hash_operation<T>(operation, input_tensors);
                } else if constexpr (
                    detail::implements_create_program_with_optional_input_tensors<T>() ||
                    detail::implements_create_mesh_workload_with_optional_input_tensors<T>()) {
                    TT_FATAL(
                        not optional_input_tensors.empty(),
                        "Non-optional input tensors not supported by {}",
                        tt::stl::get_type_name<T>());
                    return hash_operation<T>(operation, input_tensors, optional_input_tensors);
                } else {
                    static_assert(
                        tt::stl::concepts::always_false_v<T>,
                        "Operation doesn't implement create_program or create_mesh_workload");
                }
            }},
        uses_custom_program_hash_impl_{[]() -> bool {
            if constexpr (detail::implements_compute_program_hash<T>()) {
                return true;
            } else if constexpr (detail::implements_compute_program_hash_with_optional_input_tensors<T>()) {
                return true;
            } else {
                return false;
            }
        }},
        has_create_workload_method_impl_{[]() -> bool {
            // Operation must implement exactly one of the `create_program` or `create_mesh_workload` methods.
            static_assert(
                (detail::implements_create_mesh_workload<T>() ||
                 detail::implements_create_mesh_workload_with_optional_input_tensors<T>()) !=
                (detail::implements_create_program<T>() ||  //
                 detail::implements_create_program_with_optional_input_tensors<T>()));
            return detail::implements_create_mesh_workload<T>() ||
                   detail::implements_create_mesh_workload_with_optional_input_tensors<T>();
        }},
        create_profiler_info_impl_{[](const storage_t& storage, const Tensors& input_tensors) -> ProfilerInfo {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
            std::optional<std::string> preferred_name = std::string(tt::stl::get_type_name<T>());

            std::optional<std::string> parallelization_strategy = std::nullopt;
            if constexpr (detail::implements_get_parallelization_strategy<T>()) {
                parallelization_strategy = fmt::format("{}", operation.get_parallelization_strategy(input_tensors));
            }
            return {.preferred_name = preferred_name, .parallelization_strategy = parallelization_strategy};
        }},
        attributes_impl_{[](const storage_t& storage) -> tt::stl::reflection::Attributes {
            const auto& operation = *reinterpret_cast<const std::decay_t<T>*>(&storage);
            return tt::stl::reflection::get_attributes(operation);
        }} {
        static_assert(sizeof(T) <= sizeof(storage_t));
    }

    DeviceOperation(const DeviceOperation& other) :
        pointer{other.pointer ? other.copy_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        get_type_name_impl_{other.get_type_name_impl_},
        validate_impl_{other.validate_impl_},
        compute_output_specs_impl_{other.compute_output_specs_impl_},
        create_output_tensors_impl_{other.create_output_tensors_impl_},
        create_program_impl_{other.create_program_impl_},
        create_mesh_workload_impl_{other.create_mesh_workload_impl_},
        create_op_performance_model_impl_{other.create_op_performance_model_impl_},
        override_runtime_arguments_program_impl_{other.override_runtime_arguments_program_impl_},
        override_runtime_arguments_workload_impl_{other.override_runtime_arguments_workload_impl_},
        uses_custom_program_hash_impl_{other.uses_custom_program_hash_impl_},
        has_create_workload_method_impl_{other.has_create_workload_method_impl_},
        compute_program_hash_impl_{other.compute_program_hash_impl_},
        create_profiler_info_impl_{other.create_profiler_info_impl_},
        attributes_impl_{other.attributes_impl_} {}

    DeviceOperation& operator=(const DeviceOperation& other) {
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.copy_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->copy_storage = other.copy_storage;
            this->move_storage = other.move_storage;
            this->get_type_name_impl_ = other.get_type_name_impl_;
            this->validate_impl_ = other.validate_impl_;
            this->compute_output_specs_impl_ = other.compute_output_specs_impl_;
            this->create_output_tensors_impl_ = other.create_output_tensors_impl_;
            this->create_program_impl_ = other.create_program_impl_;
            this->create_mesh_workload_impl_ = other.create_mesh_workload_impl_;
            this->create_op_performance_model_impl_ = other.create_op_performance_model_impl_;
            this->override_runtime_arguments_program_impl_ = other.override_runtime_arguments_program_impl_;
            this->override_runtime_arguments_workload_impl_ = other.override_runtime_arguments_workload_impl_;
            this->uses_custom_program_hash_impl_ = other.uses_custom_program_hash_impl_;
            this->has_create_workload_method_impl_ = other.has_create_workload_method_impl_;
            this->compute_program_hash_impl_ = other.compute_program_hash_impl_;
            this->create_profiler_info_impl_ = other.create_profiler_info_impl_;
            this->attributes_impl_ = other.attributes_impl_;
        }
        return *this;
    }

    DeviceOperation(DeviceOperation&& other) noexcept :
        pointer{other.pointer ? other.move_storage(this->type_erased_storage, other.pointer) : nullptr},
        delete_storage{other.delete_storage},
        copy_storage{other.copy_storage},
        move_storage{other.move_storage},
        get_type_name_impl_{other.get_type_name_impl_},
        validate_impl_{other.validate_impl_},
        compute_output_specs_impl_{other.compute_output_specs_impl_},
        create_output_tensors_impl_{other.create_output_tensors_impl_},
        create_program_impl_{other.create_program_impl_},
        create_mesh_workload_impl_{other.create_mesh_workload_impl_},
        create_op_performance_model_impl_{other.create_op_performance_model_impl_},
        override_runtime_arguments_program_impl_{other.override_runtime_arguments_program_impl_},
        override_runtime_arguments_workload_impl_{other.override_runtime_arguments_workload_impl_},
        uses_custom_program_hash_impl_{other.uses_custom_program_hash_impl_},
        has_create_workload_method_impl_{other.has_create_workload_method_impl_},
        compute_program_hash_impl_{other.compute_program_hash_impl_},
        create_profiler_info_impl_{other.create_profiler_info_impl_},
        attributes_impl_{other.attributes_impl_} {}

    DeviceOperation& operator=(DeviceOperation&& other) noexcept {
        if (other.pointer != this->pointer) {
            this->destruct();
            this->pointer = nullptr;
            if (other.pointer) {
                this->pointer = other.move_storage(this->type_erased_storage, other.pointer);
            }
            this->delete_storage = other.delete_storage;
            this->copy_storage = other.copy_storage;
            this->move_storage = other.move_storage;
            this->get_type_name_impl_ = other.get_type_name_impl_;
            this->validate_impl_ = other.validate_impl_;
            this->compute_output_specs_impl_ = other.compute_output_specs_impl_;
            this->create_output_tensors_impl_ = other.create_output_tensors_impl_;
            this->create_program_impl_ = other.create_program_impl_;
            this->create_mesh_workload_impl_ = other.create_mesh_workload_impl_;
            this->create_op_performance_model_impl_ = other.create_op_performance_model_impl_;
            this->override_runtime_arguments_program_impl_ = other.override_runtime_arguments_program_impl_;
            this->override_runtime_arguments_workload_impl_ = other.override_runtime_arguments_workload_impl_;
            this->uses_custom_program_hash_impl_ = other.uses_custom_program_hash_impl_;
            this->has_create_workload_method_impl_ = other.has_create_workload_method_impl_;
            this->compute_program_hash_impl_ = other.compute_program_hash_impl_;
            this->create_profiler_info_impl_ = other.create_profiler_info_impl_;
            this->attributes_impl_ = other.attributes_impl_;
        }
        return *this;
    }

    ~DeviceOperation() { this->destruct(); }

private:
    alignas(32) void* pointer = nullptr;
    alignas(32) storage_t type_erased_storage;

    void (*delete_storage)(storage_t&) = nullptr;
    void* (*copy_storage)(storage_t& storage, const void*) = nullptr;
    void* (*move_storage)(storage_t& storage, void*) = nullptr;

    std::string (*get_type_name_impl_)(const storage_t& value);
    void (*validate_impl_)(
        const storage_t& value,
        const Tensors&,
        const std::vector<std::optional<const Tensor>>&,
        const OptionalTensors&);
    ComputedSpecs (*compute_output_specs_impl_)(const storage_t& value, const Tensors&, const OptionalTensors&);
    OutputTensors (*create_output_tensors_impl_)(const storage_t& value, const Tensors&, const OptionalTensors&);

    CacheableProgram<OutputTensors> (*create_program_impl_)(
        const storage_t& value, const Tensors&, const std::vector<std::optional<const Tensor>>&, OutputTensors&);

    CacheableMeshWorkload<OutputTensors> (*create_mesh_workload_impl_)(
        const storage_t& value,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const Tensors&,
        const std::vector<std::optional<const Tensor>>&,
        OutputTensors&);

    OpPerformanceModelGeneral<OutputTensors> (*create_op_performance_model_impl_)(
        const storage_t& value, const Tensors&, const std::vector<std::optional<const Tensor>>&, OutputTensors&);
    void (*override_runtime_arguments_program_impl_)(
        const storage_t& value,
        OverrideRuntimeArgumentsCallback<OutputTensors>&,
        Program&,
        const Tensors&,
        const std::vector<std::optional<const Tensor>>&,
        OutputTensors&);
    void (*override_runtime_arguments_workload_impl_)(
        const storage_t& value,
        OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>&,
        distributed::MeshWorkload&,
        const Tensors&,
        const std::vector<std::optional<const Tensor>>&,
        OutputTensors&);
    bool (*uses_custom_program_hash_impl_)();
    bool (*has_create_workload_method_impl_)();
    Hash (*compute_program_hash_impl_)(
        const storage_t& value, const Tensors&, const std::vector<std::optional<const Tensor>>&);
    ProfilerInfo (*create_profiler_info_impl_)(const storage_t& value, const Tensors& input_tensors);
    tt::stl::reflection::Attributes (*attributes_impl_)(const storage_t& value);

    void destruct() noexcept {
        if (this->pointer) {
            this->delete_storage(this->type_erased_storage);
        }
        this->pointer = nullptr;
    }
};

struct ExternalOperation {
    using OutputTensors = Tensors;
    const std::string function_name_;
    const tt::stl::reflection::Attributes attributes_;

    std::string get_type_name() const { return this->function_name_; }
    tt::stl::reflection::Attributes attributes() const { return this->attributes_; }
};

using ProgramWithCallbacks = CacheableProgram<Tensors>;
using ProgramWithOptionalOutputTensors = CacheableProgram<OptionalTensors>;

using MeshWorkloadWithCallbacks = CacheableMeshWorkload<Tensors>;
using MeshWorkloadWithOptionalOutputTensors = CacheableMeshWorkload<OptionalTensors>;

using Operation = std::variant<DeviceOperation<Tensors>, DeviceOperation<OptionalTensors>, ExternalOperation>;

}  // namespace operation
}  // namespace tt_metal
}  // namespace tt

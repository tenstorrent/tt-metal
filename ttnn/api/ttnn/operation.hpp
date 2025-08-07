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

// ─────────────────────────────────────────────────────────────────────────────
// DeviceOperation  –  function-pointer v-table, full behaviour
// ─────────────────────────────────────────────────────────────────────────────
template <class OutputTensorsT = Tensors>
class DeviceOperation final {
public:
    using storage_t = std::array<std::byte, 1240>;
    using OutputTensors = OutputTensorsT;
    using ComputedSpecs = std::vector<ttnn::TensorSpec>;

    // ── public API (unchanged) ───────────────────────────────────────────
    std::string get_type_name() const { return vtbl_->get_type_name_(buf_); }
    void validate(const Tensors& i, const OptionalConstTensors& oi, const OptionalTensors& oo) const {
        vtbl_->validate_(buf_, i, oi, oo);
    }
    ComputedSpecs compute_output_specs(const Tensors& i, const OptionalTensors& oo) const {
        return vtbl_->compute_output_specs_(buf_, i, oo);
    }
    OutputTensors create_output_tensors(const Tensors& i, const OptionalTensors& oo) const {
        return vtbl_->create_output_tensors_(buf_, i, oo);
    }
    CacheableProgram<OutputTensors> create_program(
        const Tensors& i, const OptionalConstTensors& oi, OutputTensors& o) const {
        return vtbl_->create_program_(buf_, i, oi, o);
    }
    CacheableMeshWorkload<OutputTensors> create_mesh_workload(
        const ttnn::MeshCoordinateRangeSet& coords,
        const Tensors& i,
        const OptionalConstTensors& oi,
        OutputTensors& o) const {
        return vtbl_->create_mesh_workload_(buf_, coords, i, oi, o);
    }
    bool has_create_workload_method() const { return vtbl_->has_create_workload_(); }
    OpPerformanceModelGeneral<OutputTensors> create_op_performance_model(
        const Tensors& i, const OptionalConstTensors& oi, OutputTensors& o) const {
        return vtbl_->create_op_model_(buf_, i, oi, o);
    }
    void override_runtime_arguments(
        OverrideRuntimeArgumentsCallback<OutputTensors>& cb,
        Program& p,
        const Tensors& i,
        const OptionalConstTensors& oi,
        OutputTensors& o) const {
        vtbl_->override_rt_prog_(buf_, cb, p, i, oi, o);
    }
    void override_runtime_arguments(
        OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>& cb,
        distributed::MeshWorkload& wl,
        const Tensors& i,
        const OptionalConstTensors& oi,
        OutputTensors& o) const {
        vtbl_->override_rt_wl_(buf_, cb, wl, i, oi, o);
    }
    bool uses_custom_program_hash() const { return vtbl_->uses_custom_hash_(); }
    Hash compute_program_hash(const Tensors& i, const OptionalConstTensors& oi) const {
        return vtbl_->compute_hash_(buf_, i, oi);
    }
    ProfilerInfo create_profiler_info(const Tensors& i) const { return vtbl_->create_profiler_(buf_, i); }
    tt::stl::reflection::Attributes attributes() const { return vtbl_->attributes_(buf_); }

    // perfect-forwarding ctor
    template <
        typename ConcreteOp,
        typename = std::enable_if_t<!std::is_same_v<std::decay_t<ConcreteOp>, DeviceOperation>>>
    explicit DeviceOperation(ConcreteOp&& op) {
        static_assert(sizeof(std::decay_t<ConcreteOp>) <= sizeof(storage_t), "Concrete op exceeds inline buffer");
        new (&buf_) std::decay_t<ConcreteOp>(std::forward<ConcreteOp>(op));
        vtbl_ = &make_vtbl<std::decay_t<ConcreteOp>>();
    }

    // rule-of-five via v-table helpers
    DeviceOperation(const DeviceOperation& other) {
        other.vtbl_->copy_(&buf_, &other.buf_);
        vtbl_ = other.vtbl_;
    }
    DeviceOperation(DeviceOperation&& other) noexcept {
        other.vtbl_->move_(&buf_, &other.buf_);
        vtbl_ = other.vtbl_;
        other.vtbl_ = nullptr;
    }
    DeviceOperation& operator=(const DeviceOperation& rhs) {
        if (this != &rhs) {
            destroy();
            rhs.vtbl_->copy_(&buf_, &rhs.buf_);
            vtbl_ = rhs.vtbl_;
        }
        return *this;
    }
    DeviceOperation& operator=(DeviceOperation&& rhs) noexcept {
        if (this != &rhs) {
            destroy();
            rhs.vtbl_->move_(&buf_, &rhs.buf_);
            vtbl_ = rhs.vtbl_;
            rhs.vtbl_ = nullptr;
        }
        return *this;
    }
    ~DeviceOperation() { destroy(); }

private:
    // ── v-table definition ──────────────────────────────────────────────
    struct VTable {
        std::string (*get_type_name_)(const storage_t&);
        void (*validate_)(const storage_t&, const Tensors&, const OptionalConstTensors&, const OptionalTensors&);
        ComputedSpecs (*compute_output_specs_)(const storage_t&, const Tensors&, const OptionalTensors&);
        OutputTensors (*create_output_tensors_)(const storage_t&, const Tensors&, const OptionalTensors&);
        CacheableProgram<OutputTensors> (*create_program_)(
            const storage_t&, const Tensors&, const OptionalConstTensors&, OutputTensors&);
        CacheableMeshWorkload<OutputTensors> (*create_mesh_workload_)(
            const storage_t&,
            const ttnn::MeshCoordinateRangeSet&,
            const Tensors&,
            const OptionalConstTensors&,
            OutputTensors&);
        OpPerformanceModelGeneral<OutputTensors> (*create_op_model_)(
            const storage_t&, const Tensors&, const OptionalConstTensors&, OutputTensors&);
        void (*override_rt_prog_)(
            const storage_t&,
            OverrideRuntimeArgumentsCallback<OutputTensors>&,
            Program&,
            const Tensors&,
            const OptionalConstTensors&,
            OutputTensors&);
        void (*override_rt_wl_)(
            const storage_t&,
            OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>&,
            distributed::MeshWorkload&,
            const Tensors&,
            const OptionalConstTensors&,
            OutputTensors&);
        Hash (*compute_hash_)(const storage_t&, const Tensors&, const OptionalConstTensors&);
        bool (*uses_custom_hash_)();
        bool (*has_create_workload_)();
        ProfilerInfo (*create_profiler_)(const storage_t&, const Tensors&);
        tt::stl::reflection::Attributes (*attributes_)(const storage_t&);
        // lifetime
        void* (*copy_)(storage_t*, const storage_t*);
        void* (*move_)(storage_t*, storage_t*);
        void (*destroy_)(storage_t*);
    };

    // ── v-table generator (one per concrete op) ─────────────────────────
    template <class C>
    static const VTable& make_vtbl() {
        using detail::implements_validate;
        using detail::implements_validate_with_optional_input_tensors;
        using detail::implements_validate_with_output_tensors;
        using detail::implements_validate_with_output_tensors_and_optional_input_tensors;
        /* other implement* aliases omitted for brevity */

        static const VTable vt{
            /* get_type_name_ */
            [](const storage_t& s) -> std::string {
                const C& o = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_get_type_name<C>()) {
                    return o.get_type_name();
                } else {
                    return std::string(tt::stl::get_type_name<C>());
                }
            },
            /* validate_ */
            [](const storage_t& s,
               const Tensors& in,
               const OptionalConstTensors& opt_in,
               const OptionalTensors& opt_out) {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (implements_validate<C>()) {
                    TT_FATAL(opt_in.empty(), "Optional input tensors not allowed");
                    op.validate(in);
                } else if constexpr (implements_validate_with_optional_input_tensors<C>()) {
                    TT_FATAL(!opt_in.empty(), "Optional input tensors are expected");
                    op.validate(in, opt_in);
                } else if constexpr (implements_validate_with_output_tensors<C>()) {
                    TT_FATAL(opt_in.empty(), "Optional input tensors not allowed");
                    op.validate_with_output_tensors(in, opt_out);
                } else if constexpr (implements_validate_with_output_tensors_and_optional_input_tensors<C>()) {
                    TT_FATAL(!opt_in.empty(), "Optional input tensors are expected");
                    TT_FATAL(!opt_out.empty(), "Optional output tensors are expected");
                    op.validate_with_output_tensors(in, opt_in, opt_out);
                }
            },
            /* compute_output_specs_ */
            [](const storage_t& s, const Tensors& in, const OptionalTensors& oo) -> ComputedSpecs {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_compute_output_specs_with_optional_output_tensors<C>()) {
                    return op.compute_output_specs(in, oo);
                } else {
                    return op.compute_output_specs(in);
                }
            },
            /* create_output_tensors_ */
            [](const storage_t& s, const Tensors& in, const OptionalTensors& oo) -> OutputTensors {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_create_output_tensors_with_optional_output_tensors<C>()) {
                    return op.create_output_tensors(in, oo);
                } else if constexpr (detail::implements_create_output_tensors<C>()) {
                    return op.create_output_tensors(in);
                } else {
                    return default_create_output_tensors(op, in, oo);
                }
            },
            /* create_program_ */
            [](const storage_t& s, const Tensors& in, const OptionalConstTensors& oi, OutputTensors& o)
                -> CacheableProgram<OutputTensors> {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_create_program<C>()) {
                    return op.create_program(in, o);
                } else if constexpr (detail::implements_create_program_with_optional_input_tensors<C>()) {
                    return op.create_program(in, oi, o);  // optional-input version
                }
                return {};
            },
            /* create_mesh_workload_ */
            [](const storage_t& s,
               const ttnn::MeshCoordinateRangeSet& coords,
               const Tensors& in,
               const OptionalConstTensors& oi,
               OutputTensors& o) -> CacheableMeshWorkload<OutputTensors> {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_create_mesh_workload<C>()) {
                    return op.create_mesh_workload(coords, in, o);
                } else if constexpr (detail::implements_create_mesh_workload_with_optional_input_tensors<C>()) {
                    return op.create_mesh_workload(coords, in, oi, o);
                }

                return {};
            },
            /* create_op_model_ */
            [](const storage_t& s, const Tensors& in, const OptionalConstTensors& oi, OutputTensors& o) {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_create_op_performance_model<C>()) {
                    return op.create_op_performance_model(in, oi, o);
                } else {
                    return OpPerformanceModelGeneral<OutputTensors>(in, o, 1);
                }
            },
            /* override_rt_prog_ */
            [](const storage_t& s,
               OverrideRuntimeArgumentsCallback<OutputTensors>& cb,
               Program& p,
               const Tensors& in,
               const OptionalConstTensors& oi,
               OutputTensors& o) { cb(&*reinterpret_cast<const C*>(&s), p, in, oi, o); },
            /* override_rt_wl_ */
            [](const storage_t& s,
               OverrideRuntimeArgumentsWorkloadCallback<OutputTensors>& cb,
               distributed::MeshWorkload& wl,
               const Tensors& in,
               const OptionalConstTensors& oi,
               OutputTensors& o) { cb(&*reinterpret_cast<const C*>(&s), wl, in, oi, o); },
            /* compute_hash_ */
            [](const storage_t& s, const Tensors& in, const OptionalConstTensors& oi) -> Hash {
                const C& op = *reinterpret_cast<const C*>(&s);
                if constexpr (detail::implements_compute_program_hash<C>()) {
                    return op.compute_program_hash(in);
                } else if constexpr (detail::implements_compute_program_hash_with_optional_input_tensors<C>()) {
                    return op.compute_program_hash(in, oi);
                } else if constexpr (
                    detail::implements_create_program<C>() || detail::implements_create_mesh_workload<C>()) {
                    return hash_operation<C>(op, in);
                } else {
                    return hash_operation<C>(op, in, oi);
                }
            },
            /* uses_custom_hash_   */
            []() {
                return detail::implements_compute_program_hash<C>() ||
                       detail::implements_compute_program_hash_with_optional_input_tensors<C>();
            },
            /* has_create_workload_*/
            []() {
                return detail::implements_create_mesh_workload<C>() ||
                       detail::implements_create_mesh_workload_with_optional_input_tensors<C>();
            },
            /* create_profiler_    */
            [](const storage_t& s, const Tensors& in) {
                const C& op = *reinterpret_cast<const C*>(&s);
                ProfilerInfo pi{std::string(tt::stl::get_type_name<C>()), std::nullopt};
                if constexpr (detail::implements_get_parallelization_strategy<C>()) {
                    pi.parallelization_strategy = fmt::format("{}", op.get_parallelization_strategy(in));
                }
                return pi;
            },
            /* attributes_         */
            [](const storage_t& s) {
                const C& op = *reinterpret_cast<const C*>(&s);
                return tt::stl::reflection::get_attributes(op);
            },
            /* copy_               */
            [](storage_t* dst, const storage_t* src) -> void* { return new (dst) C(*reinterpret_cast<const C*>(src)); },
            /* move_               */
            [](storage_t* dst, storage_t* src) -> void* { return new (dst) C(std::move(*reinterpret_cast<C*>(src))); },
            /* destroy_            */ [](storage_t* s) { reinterpret_cast<C*>(s)->~C(); }};
        return vt;
    }

    void destroy() noexcept {
        if (vtbl_) {
            vtbl_->destroy_(&buf_);
            vtbl_ = nullptr;
        }
    }

    storage_t buf_{};
    const VTable* vtbl_ = nullptr;
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

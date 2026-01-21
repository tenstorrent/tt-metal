// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <experimental/type_traits>
#include <ttnn/tensor/tensor.hpp>
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/config.hpp"
#include "ttnn/distributed/types.hpp"

#include <tracy/Tracy.hpp>

namespace tt::tt_metal::operation {

using Hash = tt::stl::hash::hash_t;

template <typename OperationType, typename... Types>
static Hash hash_operation(const Types&... objects) {
    return stl::hash::hash_objects_with_default_seed(tt::stl::hash::type_hash<OperationType>, objects...);
}

using Tensors = std::vector<Tensor>;
using OptionalTensors = std::vector<std::optional<Tensor>>;
using OptionalConstTensors = std::vector<std::optional<const Tensor>>;

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
    std::vector<int> inputs_bytes;
    std::vector<int> outputs_bytes;

    OpPerformanceModelGeneral(Tensors input_tensors, const OutputTensors& output_tensors, int ideal_compute_cycles) :
        ideal_compute_cycles(ideal_compute_cycles) {
        const auto& t = input_tensors.at(0);
        const auto arch = t.storage_type() == StorageType::DEVICE ? t.device()->arch() : ARCH::WORMHOLE_B0;

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

        auto tensor_ns = [peak_dram_bw](const Tensor& t) {
            int size_bytes = t.physical_volume() * t.element_size();
            if (t.memory_config().is_dram()) {
                return size_bytes / peak_dram_bw / 1024 / 1024 / 1024 * 1000 * 1000 * 1000;
            }
            if (t.memory_config().is_l1()) {
                return 1.0f;  // TODO: figure out better modelling scheme for L1->L1 Transfers
                              // return size_bytes / noc_l1_bisection_bw / 1024 / 1024 / 1024 * 1000 * 1000 *
                              // 1000;
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
        switch (f) {
            case MathFidelity::Invalid: return 0;
            case MathFidelity::LoFi: return 1;
            case MathFidelity::HiFi2: return 2;
            case MathFidelity::HiFi3: return 3;
            case MathFidelity::HiFi4: return 4;
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

}  // namespace detail

struct ExternalOperation {
    using OutputTensors = Tensors;
    const std::string function_name_;
    const tt::stl::reflection::Attributes attributes_;

    std::string get_type_name() const { return this->function_name_; }
    tt::stl::reflection::Attributes attributes() const { return this->attributes_; }
};

}  // namespace tt::tt_metal::operation

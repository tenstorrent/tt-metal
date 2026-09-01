// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_performance_model.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>

#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim::kda_performance_model {
namespace {

constexpr double PROFILER_INT_MAX = std::numeric_limits<int>::max();
// Matrix capacity: tech_reports/matrix_engine/matrix_engine.md.
constexpr double MATRIX_FLOPS_PER_CORE_CYCLE = 4096.0;
// Blackhole ceiling used by the canonical operation model: ttnn/core/operation.cpp.
constexpr double DRAM_BYTES_PER_NS = 512.0;

int narrow_profiler_int(std::string_view quantity, double value) {
    TT_FATAL(
        std::isfinite(value) && value >= 0.0 && value <= PROFILER_INT_MAX,
        "KDA performance-model {} must be finite and fit profiler int",
        quantity);
    return static_cast<int>(value);
}

double fidelity_factor(tt::tt_metal::MathFidelity fidelity) {
    switch (fidelity) {
        case tt::tt_metal::MathFidelity::LoFi: return 1.0;
        case tt::tt_metal::MathFidelity::HiFi2: return 2.0;
        case tt::tt_metal::MathFidelity::HiFi3: return 3.0;
        case tt::tt_metal::MathFidelity::HiFi4: return 4.0;
        case tt::tt_metal::MathFidelity::Invalid:
            TT_FATAL(false, "KDA performance model received invalid math fidelity");
    }
    TT_FATAL(false, "KDA performance model received unsupported math fidelity");
    return 0.0;
}

double physical_bytes(const Tensor& tensor) {
    return static_cast<double>(tensor.physical_volume()) * tensor.element_size();
}

void validate_tensor(const Tensor& tensor, tt::tt_metal::IDevice* device, std::string_view role) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated() && tensor.buffer() != nullptr,
        "KDA performance model requires allocated device {} tensors",
        role);
    TT_FATAL(tensor.device() == device, "KDA performance model requires all {} tensors on the same device", role);
}

}  // namespace

KdaProfilerModel make_profiler_model(
    const KdaFpuWork& work,
    std::span<const Tensor* const> inputs,
    const std::vector<Tensor>& outputs,
    tt::tt_metal::MathFidelity math_fidelity) {
    TT_FATAL(!inputs.empty(), "KDA performance model requires at least one input tensor");
    TT_FATAL(inputs.front() != nullptr, "KDA performance model received a null input tensor");
    const Tensor& first_input = *inputs.front();
    TT_FATAL(
        first_input.storage_type() == StorageType::DEVICE && first_input.is_allocated() &&
            first_input.buffer() != nullptr && first_input.device() != nullptr,
        "KDA performance model requires allocated device input tensors");

    auto* device = first_input.device();
    TT_FATAL(device->arch() == tt::ARCH::BLACKHOLE, "KDA performance model supports Blackhole only");

    KdaProfilerModel result;
    result.inputs_bytes.assign(inputs.size(), 0);
    result.outputs_bytes.assign(outputs.size(), 0);

    double mandatory_dram_bytes = 0.0;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        TT_FATAL(inputs[index] != nullptr, "KDA performance model received a null input tensor");
        const Tensor& input = *inputs[index];
        validate_tensor(input, device, "input");
        if (input.memory_config().is_dram()) {
            const double bytes = physical_bytes(input);
            result.inputs_bytes[index] = narrow_profiler_int("input bytes", bytes);
            mandatory_dram_bytes += bytes;
        }
    }

    for (std::size_t index = 0; index < outputs.size(); ++index) {
        const Tensor& output = outputs[index];
        validate_tensor(output, device, "output");
        if (output.memory_config().is_dram()) {
            const double bytes = physical_bytes(output);
            result.outputs_bytes[index] = narrow_profiler_int("output bytes", bytes);
            mandatory_dram_bytes += bytes;
        }
    }

    const double factor = fidelity_factor(math_fidelity);
    const auto grid = device->compute_with_storage_grid_size();
    const double core_count = static_cast<double>(grid.x) * grid.y;
    TT_FATAL(core_count > 0.0, "KDA performance model requires at least one compute core");
    const int clock_mhz = device->get_clock_rate_mhz();
    TT_FATAL(clock_mhz > 0, "KDA performance model requires a positive device clock");

    const double cycle_numerator = work.fpu_matrix_flops * factor + 32.0 * work.fpu_multiply_ops * factor +
                                   32.0 * work.fpu_add_ops + 16.0 * work.fpu_reduction_ops * factor;
    const double ideal_fpu_cycles = std::ceil(cycle_numerator / (MATRIX_FLOPS_PER_CORE_CYCLE * core_count));
    const double ideal_fpu_ns = std::ceil(ideal_fpu_cycles * 1000.0 / clock_mhz);
    const double ideal_dram_ns = std::ceil(mandatory_dram_bytes / DRAM_BYTES_PER_NS);
    const double ideal_ns = std::max(ideal_fpu_ns, ideal_dram_ns);

    result.ideal_compute_cycles = narrow_profiler_int("ideal compute cycles", std::max(1.0, ideal_fpu_cycles));
    result.ideal_compute_ns = narrow_profiler_int("ideal compute ns", std::max(1.0, ideal_fpu_ns));
    result.ideal_bandwidth_ns = narrow_profiler_int("ideal bandwidth ns", std::max(1.0, ideal_dram_ns));
    result.ideal_ns = narrow_profiler_int("ideal ns", std::max(1.0, ideal_ns));
    return result;
}

}  // namespace ttnn::experimental::prim::kda_performance_model

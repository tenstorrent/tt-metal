// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_performance_model.hpp"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <string_view>
#include <unordered_set>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim::kda_performance_model {
namespace {

using Wide = unsigned __int128;

constexpr Wide U64_MAX = std::numeric_limits<uint64_t>::max();
constexpr Wide WIDE_MAX = ~Wide{0};
constexpr Wide PROFILER_INT_MAX = std::numeric_limits<int>::max();
// Matrix capacity: tech_reports/matrix_engine/matrix_engine.md.
constexpr uint64_t MATRIX_FLOPS_PER_CORE_CYCLE = 4096;
// Blackhole ceiling used by the canonical operation model: ttnn/core/operation.cpp.
constexpr uint64_t DRAM_BYTES_PER_NS = 512;

Wide checked_sum(std::string_view operation, std::string_view quantity, std::initializer_list<Wide> terms) {
    Wide result = 0;
    for (const Wide term : terms) {
        TT_FATAL(term <= WIDE_MAX - result, "KDA {} performance-model {} overflowed while adding", operation, quantity);
        result += term;
    }
    return result;
}

Wide checked_product(std::string_view operation, std::string_view quantity, std::initializer_list<Wide> factors) {
    Wide result = 1;
    for (const Wide factor : factors) {
        TT_FATAL(
            result == 0 || factor <= WIDE_MAX / result,
            "KDA {} performance-model {} overflowed while multiplying",
            operation,
            quantity);
        result *= factor;
    }
    return result;
}

uint64_t narrow_u64(std::string_view operation, std::string_view quantity, Wide value) {
    TT_FATAL(value <= U64_MAX, "KDA {} performance-model {} does not fit uint64", operation, quantity);
    return static_cast<uint64_t>(value);
}

int narrow_profiler_int(std::string_view quantity, Wide value) {
    TT_FATAL(value <= PROFILER_INT_MAX, "KDA performance-model {} does not fit profiler int", quantity);
    return static_cast<int>(value);
}

uint64_t fidelity_factor(tt::tt_metal::MathFidelity fidelity) {
    switch (fidelity) {
        case tt::tt_metal::MathFidelity::LoFi: return 1;
        case tt::tt_metal::MathFidelity::HiFi2: return 2;
        case tt::tt_metal::MathFidelity::HiFi3: return 3;
        case tt::tt_metal::MathFidelity::HiFi4: return 4;
        case tt::tt_metal::MathFidelity::Invalid:
            TT_FATAL(false, "KDA performance model received invalid math fidelity");
    }
    TT_FATAL(false, "KDA performance model received unsupported math fidelity");
    return 0;
}

Wide ceil_div(Wide numerator, Wide denominator) {
    TT_FATAL(denominator > 0, "KDA performance-model divisor must be positive");
    return numerator / denominator + static_cast<Wide>(numerator % denominator != 0);
}

Wide physical_bytes(const Tensor& tensor, std::string_view role) {
    return checked_product("tensor traffic", role, {Wide{tensor.physical_volume()}, Wide{tensor.element_size()}});
}

void validate_tensor(const Tensor& tensor, tt::tt_metal::IDevice* device, std::string_view role) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated() && tensor.buffer() != nullptr,
        "KDA performance model requires allocated device {} tensors",
        role);
    TT_FATAL(tensor.device() == device, "KDA performance model requires all {} tensors on the same device", role);
}

}  // namespace

KdaFpuWork sigmoid_gated_rms_norm_work(uint64_t batch, uint64_t num_heads, uint64_t sequence, uint64_t value_dim) {
    constexpr std::string_view operation = "sigmoid_gated_rms_norm";
    TT_FATAL(value_dim > 0, "KDA {} performance model requires positive value_dim", operation);
    const Wide rows = checked_product(operation, "rows", {Wide{batch}, Wide{num_heads}, Wide{sequence}});
    const Wide elements = checked_product(operation, "elements", {rows, Wide{value_dim}});
    return KdaFpuWork{
        .fpu_multiply_ops = narrow_u64(
            operation,
            "FPU multiply operations",
            checked_product(operation, "FPU multiply operations", {Wide{4}, elements})),
        .fpu_add_ops = narrow_u64(operation, "FPU add operations", rows),
        .fpu_reduction_ops = narrow_u64(
            operation,
            "FPU reduction operations",
            checked_product(operation, "FPU reduction operations", {rows, Wide{value_dim - 1}})),
    };
}

KdaFpuWork qkv_causal_conv1d_silu_work(
    uint64_t batch, uint64_t sequence, uint64_t q_width, uint64_t k_width, uint64_t v_width) {
    constexpr std::string_view operation = "qkv_causal_conv1d_silu";
    const Wide width = checked_sum(operation, "width", {Wide{q_width}, Wide{k_width}, Wide{v_width}});
    const Wide elements = checked_product(operation, "elements", {Wide{batch}, Wide{sequence}, width});
    return KdaFpuWork{
        .fpu_multiply_ops = narrow_u64(
            operation,
            "FPU multiply operations",
            checked_product(operation, "FPU multiply operations", {Wide{4}, elements})),
        .fpu_add_ops = narrow_u64(
            operation, "FPU add operations", checked_product(operation, "FPU add operations", {Wide{3}, elements})),
    };
}

KdaFpuWork reduce_affine_transforms_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim) {
    constexpr std::string_view operation = "reduce_affine_transforms";
    TT_FATAL(groups_per_head > 0, "KDA {} performance model requires positive groups_per_head", operation);
    const Wide compositions =
        checked_product(operation, "compositions", {Wide{batch_heads}, Wide{groups_per_head - 1}});
    const Wide key_squared = checked_product(operation, "key squared", {Wide{key_dim}, Wide{key_dim}});
    const Wide matrix_flops_per_composition = checked_sum(
        operation,
        "matrix FLOPs per composition",
        {checked_product(operation, "matrix FLOPs per composition", {Wide{2}, key_squared, Wide{key_dim}}),
         checked_product(operation, "matrix FLOPs per composition", {Wide{2}, key_squared, Wide{value_dim}})});
    return KdaFpuWork{
        .fpu_matrix_flops = narrow_u64(
            operation,
            "matrix FLOPs",
            checked_product(operation, "matrix FLOPs", {compositions, matrix_flops_per_composition})),
        .fpu_add_ops = narrow_u64(
            operation,
            "FPU add operations",
            checked_product(operation, "FPU add operations", {compositions, Wide{key_dim}, Wide{value_dim}})),
    };
}

KdaFpuWork affine_exclusive_scan_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim) {
    constexpr std::string_view operation = "affine_exclusive_scan";
    TT_FATAL(groups_per_head > 0, "KDA {} performance model requires positive groups_per_head", operation);
    const Wide transitions = checked_product(operation, "transitions", {Wide{batch_heads}, Wide{groups_per_head - 1}});
    return KdaFpuWork{
        .fpu_matrix_flops = narrow_u64(
            operation,
            "matrix FLOPs",
            checked_product(
                operation, "matrix FLOPs", {transitions, Wide{2}, Wide{key_dim}, Wide{key_dim}, Wide{value_dim}})),
        .fpu_add_ops = narrow_u64(
            operation,
            "FPU add operations",
            checked_product(operation, "FPU add operations", {transitions, Wide{key_dim}, Wide{value_dim}})),
    };
}

KdaFpuWork prepare_chunk_recurrence_work(
    uint64_t num_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim) {
    constexpr std::string_view operation = "prepare_chunk_recurrence";
    TT_FATAL(key_dim > 0, "KDA {} performance model requires positive key_dim", operation);
    constexpr Wide chunk = tt::constants::TILE_HEIGHT;
    constexpr Wide inverse_flops = chunk * (chunk - 1) * (chunk + 1) / 3;
    const Wide instances = checked_product(operation, "instances", {Wide{num_heads}, Wide{num_chunks}});
    const Wide multiply_per_instance = checked_sum(
        operation,
        "FPU multiply operations per instance",
        {checked_product(operation, "FPU multiply operations per instance", {Wide{10}, chunk, Wide{key_dim}}),
         checked_product(operation, "FPU multiply operations per instance", {chunk, Wide{value_dim}})});
    const Wide add_per_instance = checked_sum(
        operation,
        "FPU add operations per instance",
        {checked_product(operation, "FPU add operations per instance", {Wide{2}, chunk}),
         checked_product(operation, "FPU add operations per instance", {chunk - 1, Wide{key_dim}}),
         checked_product(operation, "FPU add operations per instance", {chunk, Wide{key_dim}}),
         checked_product(operation, "FPU add operations per instance", {chunk, chunk})});
    const Wide matrix_flops_per_instance = checked_sum(
        operation,
        "matrix FLOPs per instance",
        {checked_product(operation, "matrix FLOPs per instance", {Wide{4}, chunk, chunk, Wide{key_dim}}),
         inverse_flops});
    return KdaFpuWork{
        .fpu_matrix_flops = narrow_u64(
            operation,
            "matrix FLOPs",
            checked_product(operation, "matrix FLOPs", {instances, matrix_flops_per_instance})),
        .fpu_multiply_ops = narrow_u64(
            operation,
            "FPU multiply operations",
            checked_product(operation, "FPU multiply operations", {instances, multiply_per_instance})),
        .fpu_add_ops = narrow_u64(
            operation,
            "FPU add operations",
            checked_product(operation, "FPU add operations", {instances, add_per_instance})),
        .fpu_reduction_ops = narrow_u64(
            operation,
            "FPU reduction operations",
            checked_product(operation, "FPU reduction operations", {instances, Wide{2}, chunk, Wide{key_dim - 1}})),
    };
}

KdaFpuWork recurrent_chunk_scan_work(uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim) {
    constexpr std::string_view operation = "recurrent_chunk_scan";
    constexpr Wide chunk = tt::constants::TILE_HEIGHT;
    const Wide instances = checked_product(operation, "instances", {Wide{batch_heads}, Wide{num_chunks}});
    const Wide matrix_flops_per_instance = checked_sum(
        operation,
        "matrix FLOPs per instance",
        {checked_product(operation, "matrix FLOPs per instance", {Wide{6}, chunk, Wide{key_dim}, Wide{value_dim}}),
         checked_product(operation, "matrix FLOPs per instance", {Wide{4}, chunk, chunk, Wide{value_dim}})});
    const Wide add_per_instance = checked_sum(
        operation,
        "FPU add operations per instance",
        {checked_product(operation, "FPU add operations per instance", {Wide{2}, chunk, Wide{value_dim}}),
         checked_product(operation, "FPU add operations per instance", {Wide{key_dim}, Wide{value_dim}})});
    return KdaFpuWork{
        .fpu_matrix_flops = narrow_u64(
            operation,
            "matrix FLOPs",
            checked_product(operation, "matrix FLOPs", {instances, matrix_flops_per_instance})),
        .fpu_multiply_ops = narrow_u64(
            operation,
            "FPU multiply operations",
            checked_product(operation, "FPU multiply operations", {instances, Wide{key_dim}, Wide{value_dim}})),
        .fpu_add_ops = narrow_u64(
            operation,
            "FPU add operations",
            checked_product(operation, "FPU add operations", {instances, add_per_instance})),
    };
}

KdaFpuWork summarize_chunk_recurrence_work(
    uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim) {
    constexpr std::string_view operation = "summarize_chunk_recurrence";
    constexpr Wide chunk = tt::constants::TILE_HEIGHT;
    const Wide instances = checked_product(operation, "instances", {Wide{batch_heads}, Wide{num_chunks}});
    const Wide matrix_flops_per_instance = checked_sum(
        operation,
        "matrix FLOPs per instance",
        {checked_product(operation, "matrix FLOPs per instance", {Wide{8}, chunk, Wide{key_dim}, Wide{value_dim}}),
         checked_product(operation, "matrix FLOPs per instance", {Wide{4}, chunk, chunk, Wide{value_dim}})});
    const Wide add_per_instance = checked_sum(
        operation,
        "FPU add operations per instance",
        {checked_product(operation, "FPU add operations per instance", {Wide{2}, chunk, Wide{value_dim}}),
         checked_product(operation, "FPU add operations per instance", {Wide{2}, Wide{key_dim}, Wide{value_dim}})});
    const Wide total_add_ops = checked_sum(
        operation,
        "FPU add operations",
        {checked_product(operation, "FPU add operations", {instances, add_per_instance}),
         checked_product(operation, "FPU add operations", {Wide{batch_heads}, Wide{key_dim}, Wide{value_dim}})});
    return KdaFpuWork{
        .fpu_matrix_flops = narrow_u64(
            operation,
            "matrix FLOPs",
            checked_product(operation, "matrix FLOPs", {instances, matrix_flops_per_instance})),
        .fpu_multiply_ops = narrow_u64(
            operation,
            "FPU multiply operations",
            checked_product(
                operation, "FPU multiply operations", {instances, Wide{2}, Wide{key_dim}, Wide{value_dim}})),
        .fpu_add_ops = narrow_u64(operation, "FPU add operations", total_add_ops),
    };
}

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

    Wide mandatory_dram_bytes = 0;
    std::unordered_set<const void*> input_buffers;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        TT_FATAL(inputs[index] != nullptr, "KDA performance model received a null input tensor");
        const Tensor& input = *inputs[index];
        validate_tensor(input, device, "input");
        TT_FATAL(
            input_buffers.insert(static_cast<const void*>(input.buffer())).second,
            "KDA performance model does not support aliased inputs");
        if (input.memory_config().is_dram()) {
            const Wide bytes = physical_bytes(input, "input bytes");
            result.inputs_bytes[index] = narrow_profiler_int("input bytes", bytes);
            mandatory_dram_bytes = checked_sum("tensor traffic", "mandatory DRAM bytes", {mandatory_dram_bytes, bytes});
        }
    }

    for (std::size_t index = 0; index < outputs.size(); ++index) {
        const Tensor& output = outputs[index];
        validate_tensor(output, device, "output");
        if (output.memory_config().is_dram()) {
            const Wide bytes = physical_bytes(output, "output bytes");
            result.outputs_bytes[index] = narrow_profiler_int("output bytes", bytes);
            mandatory_dram_bytes = checked_sum("tensor traffic", "mandatory DRAM bytes", {mandatory_dram_bytes, bytes});
        }
    }

    const uint64_t factor = fidelity_factor(math_fidelity);
    const auto grid = device->compute_with_storage_grid_size();
    const Wide core_count = checked_product("device", "compute core count", {Wide{grid.x}, Wide{grid.y}});
    TT_FATAL(core_count > 0, "KDA performance model requires at least one compute core");
    const int clock_mhz = device->get_clock_rate_mhz();
    TT_FATAL(clock_mhz > 0, "KDA performance model requires a positive device clock");

    const Wide cycle_numerator = checked_sum(
        "FPU estimate",
        "cycle numerator",
        {checked_product("FPU estimate", "matrix cycle numerator", {Wide{work.fpu_matrix_flops}, Wide{factor}}),
         checked_product(
             "FPU estimate", "multiply cycle numerator", {Wide{32}, Wide{work.fpu_multiply_ops}, Wide{factor}}),
         checked_product("FPU estimate", "add cycle numerator", {Wide{32}, Wide{work.fpu_add_ops}}),
         checked_product(
             "FPU estimate", "reduction cycle numerator", {Wide{16}, Wide{work.fpu_reduction_ops}, Wide{factor}})});
    const Wide cycle_denominator =
        checked_product("FPU estimate", "cycle denominator", {Wide{MATRIX_FLOPS_PER_CORE_CYCLE}, core_count});
    const Wide ideal_fpu_cycles = ceil_div(cycle_numerator, cycle_denominator);
    const Wide ideal_fpu_ns =
        ceil_div(checked_product("FPU estimate", "time numerator", {ideal_fpu_cycles, Wide{1000}}), Wide{clock_mhz});
    const Wide ideal_dram_ns = ceil_div(mandatory_dram_bytes, Wide{DRAM_BYTES_PER_NS});
    const Wide ideal_ns = std::max(ideal_fpu_ns, ideal_dram_ns);

    result.ideal_compute_cycles = narrow_profiler_int("ideal compute cycles", std::max(Wide{1}, ideal_fpu_cycles));
    result.ideal_compute_ns = narrow_profiler_int("ideal compute ns", std::max(Wide{1}, ideal_fpu_ns));
    result.ideal_bandwidth_ns = narrow_profiler_int("ideal bandwidth ns", std::max(Wide{1}, ideal_dram_ns));
    result.ideal_ns = narrow_profiler_int("ideal ns", std::max(Wide{1}, ideal_ns));
    return result;
}

}  // namespace ttnn::experimental::prim::kda_performance_model

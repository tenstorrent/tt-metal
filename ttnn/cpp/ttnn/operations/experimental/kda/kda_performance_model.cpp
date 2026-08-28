// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_performance_model.hpp"

#include <initializer_list>
#include <limits>
#include <string_view>
#include <unordered_set>

#include <tt-metalium/constants.hpp>
#include <tt-logger/tt-logger.hpp>

namespace ttnn::experimental::prim::kda_performance_model {
namespace {

using Wide = unsigned __int128;

constexpr Wide U64_MAX = std::numeric_limits<uint64_t>::max();
constexpr Wide WIDE_MAX = ~Wide{0};
constexpr Wide PROFILER_INT_MAX = std::numeric_limits<int>::max();
// Matrix capacity: tech_reports/matrix_engine/matrix_engine.md.
constexpr uint64_t MATRIX_FLOPS_PER_CORE_CYCLE = 4096;
// Aggregate Blackhole DRAM capability:
// https://github.com/tenstorrent/tt-low-level-documentation/blob/main/data_movement_doc/general/ideal_performance.md
constexpr uint64_t DRAM_BYTES_PER_NS = 512;

std::optional<Wide> checked_sum(std::initializer_list<std::optional<Wide>> terms) {
    Wide result = 0;
    for (const auto& term : terms) {
        if (!term || *term > WIDE_MAX - result) {
            return std::nullopt;
        }
        result += *term;
    }
    return result;
}

std::optional<Wide> checked_product(std::initializer_list<std::optional<Wide>> factors) {
    Wide result = 1;
    for (const auto& factor : factors) {
        if (!factor || (result != 0 && *factor > WIDE_MAX / result)) {
            return std::nullopt;
        }
        result *= *factor;
    }
    return result;
}

std::optional<uint64_t> narrow_u64(Wide value) {
    if (value > U64_MAX) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(value);
}

std::optional<KdaWork> make_work(
    std::string_view operation,
    std::optional<Wide> dense_flops,
    std::optional<Wide> multiply_results,
    std::optional<Wide> add_results,
    std::optional<Wide> reduction_input_elements,
    std::optional<Wide> omitted_sfpu_results) {
    if (!dense_flops || !multiply_results || !add_results || !reduction_input_elements || !omitted_sfpu_results) {
        log_warning(
            tt::LogOp, "KDA {} performance-model work arithmetic overflowed; returning a zero estimate", operation);
        return std::nullopt;
    }
    const auto dense = narrow_u64(*dense_flops);
    const auto multiply = narrow_u64(*multiply_results);
    const auto add = narrow_u64(*add_results);
    const auto reduction = narrow_u64(*reduction_input_elements);
    const auto omitted = narrow_u64(*omitted_sfpu_results);
    if (!dense || !multiply || !add || !reduction || !omitted) {
        log_warning(
            tt::LogOp, "KDA {} performance-model work does not fit uint64; returning a zero estimate", operation);
        return std::nullopt;
    }
    return KdaWork{
        .dense_flops = *dense,
        .multiply_results = *multiply,
        .add_results = *add,
        .reduction_input_elements = *reduction,
        .omitted_sfpu_results = *omitted,
    };
}

std::optional<uint64_t> fidelity_factor(tt::tt_metal::MathFidelity fidelity) {
    switch (fidelity) {
        case tt::tt_metal::MathFidelity::LoFi: return 1;
        case tt::tt_metal::MathFidelity::HiFi2: return 2;
        case tt::tt_metal::MathFidelity::HiFi3: return 3;
        case tt::tt_metal::MathFidelity::HiFi4: return 4;
        case tt::tt_metal::MathFidelity::Invalid: return std::nullopt;
    }
    return std::nullopt;
}

Wide ceil_div(Wide numerator, Wide denominator) {
    return numerator / denominator + static_cast<Wide>(numerator % denominator != 0);
}

bool fits_profiler_int(Wide value) { return value <= PROFILER_INT_MAX; }

}  // namespace

std::optional<KdaWork> sigmoid_gated_rms_norm_work(
    uint64_t batch, uint64_t num_heads, uint64_t sequence, uint64_t value_dim) {
    const auto rows = checked_product({Wide{batch}, Wide{num_heads}, Wide{sequence}});
    const auto elements = checked_product({rows, Wide{value_dim}});
    return make_work(
        "sigmoid_gated_rms_norm",
        Wide{0},
        checked_product({Wide{4}, elements}),
        rows,
        elements,
        checked_sum({rows, elements}));
}

std::optional<KdaWork> qkv_causal_conv1d_silu_work(
    uint64_t batch, uint64_t sequence, uint64_t q_width, uint64_t k_width, uint64_t v_width) {
    const auto width = checked_sum({Wide{q_width}, Wide{k_width}, Wide{v_width}});
    const auto elements = checked_product({Wide{batch}, Wide{sequence}, width});
    return make_work(
        "qkv_causal_conv1d_silu",
        Wide{0},
        checked_product({Wide{4}, elements}),
        checked_product({Wide{3}, elements}),
        Wide{0},
        elements);
}

std::optional<KdaWork> reduce_affine_transforms_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim) {
    const auto compositions =
        checked_product({Wide{batch_heads}, Wide{groups_per_head == 0 ? 0 : groups_per_head - 1}});
    const auto key_squared = checked_product({Wide{key_dim}, Wide{key_dim}});
    const auto dense_per_composition = checked_sum(
        {checked_product({Wide{2}, key_squared, Wide{key_dim}}),
         checked_product({Wide{2}, key_squared, Wide{value_dim}})});
    return make_work(
        "reduce_affine_transforms",
        checked_product({compositions, dense_per_composition}),
        Wide{0},
        checked_product({compositions, Wide{key_dim}, Wide{value_dim}}),
        Wide{0},
        Wide{0});
}

std::optional<KdaWork> affine_exclusive_scan_work(
    uint64_t batch_heads, uint64_t groups_per_head, uint64_t key_dim, uint64_t value_dim) {
    const auto transitions = checked_product({Wide{batch_heads}, Wide{groups_per_head == 0 ? 0 : groups_per_head - 1}});
    return make_work(
        "affine_exclusive_scan",
        checked_product({transitions, Wide{2}, Wide{key_dim}, Wide{key_dim}, Wide{value_dim}}),
        Wide{0},
        checked_product({transitions, Wide{key_dim}, Wide{value_dim}}),
        Wide{0},
        Wide{0});
}

std::optional<KdaWork> prepare_chunk_recurrence_work(
    uint64_t num_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim) {
    constexpr Wide chunk = tt::constants::TILE_HEIGHT;
    constexpr Wide inverse_flops = chunk * (chunk - 1) * (chunk + 1) / 3;
    const auto instances = checked_product({Wide{num_heads}, Wide{num_chunks}});
    const auto multiply_per_instance =
        checked_sum({checked_product({Wide{10}, chunk, Wide{key_dim}}), checked_product({chunk, Wide{value_dim}})});
    const auto add_per_instance = checked_sum(
        {checked_product({Wide{2}, chunk}),
         checked_product({chunk - 1, Wide{key_dim}}),
         checked_product({chunk, Wide{key_dim}}),
         checked_product({chunk, chunk})});
    const auto dense_per_instance =
        checked_sum({checked_product({Wide{4}, chunk, chunk, Wide{key_dim}}), inverse_flops});
    const auto omitted_per_instance = checked_sum(
        {checked_product({Wide{2}, chunk}), checked_product({Wide{3}, chunk, Wide{key_dim}}), Wide{key_dim}});
    return make_work(
        "prepare_chunk_recurrence",
        checked_product({instances, dense_per_instance}),
        checked_product({instances, multiply_per_instance}),
        checked_product({instances, add_per_instance}),
        checked_product({instances, Wide{2}, chunk, Wide{key_dim}}),
        checked_product({instances, omitted_per_instance}));
}

std::optional<KdaWork> recurrent_chunk_scan_work(
    uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim) {
    constexpr Wide chunk = tt::constants::TILE_HEIGHT;
    const auto instances = checked_product({Wide{batch_heads}, Wide{num_chunks}});
    const auto dense_per_instance = checked_sum(
        {checked_product({Wide{6}, chunk, Wide{key_dim}, Wide{value_dim}}),
         checked_product({Wide{4}, chunk, chunk, Wide{value_dim}})});
    const auto add_per_instance = checked_sum(
        {checked_product({Wide{2}, chunk, Wide{value_dim}}), checked_product({Wide{key_dim}, Wide{value_dim}})});
    return make_work(
        "recurrent_chunk_scan",
        checked_product({instances, dense_per_instance}),
        checked_product({instances, Wide{key_dim}, Wide{value_dim}}),
        checked_product({instances, add_per_instance}),
        Wide{0},
        Wide{0});
}

std::optional<KdaWork> summarize_chunk_recurrence_work(
    uint64_t batch_heads, uint64_t num_chunks, uint64_t key_dim, uint64_t value_dim) {
    constexpr Wide chunk = tt::constants::TILE_HEIGHT;
    const auto instances = checked_product({Wide{batch_heads}, Wide{num_chunks}});
    const auto dense_per_instance = checked_sum(
        {checked_product({Wide{8}, chunk, Wide{key_dim}, Wide{value_dim}}),
         checked_product({Wide{4}, chunk, chunk, Wide{value_dim}})});
    const auto add_per_instance = checked_sum(
        {checked_product({Wide{2}, chunk, Wide{value_dim}}),
         checked_product({Wide{2}, Wide{key_dim}, Wide{value_dim}})});
    return make_work(
        "summarize_chunk_recurrence",
        checked_product({instances, dense_per_instance}),
        checked_product({instances, Wide{2}, Wide{key_dim}, Wide{value_dim}}),
        checked_sum(
            {checked_product({instances, add_per_instance}),
             checked_product({Wide{batch_heads}, Wide{key_dim}, Wide{value_dim}})}),
        Wide{0},
        Wide{0});
}

std::optional<KdaTensorTraffic> tensor_traffic(const Tensor& tensor) {
    if (tensor.storage_type() != StorageType::DEVICE || tensor.buffer() == nullptr) {
        log_warning(tt::LogOp, "KDA performance model expected an allocated device tensor; returning a zero estimate");
        return std::nullopt;
    }
    const auto byte_product = checked_product({Wide{tensor.physical_volume()}, Wide{tensor.element_size()}});
    const auto bytes = byte_product ? narrow_u64(*byte_product) : std::nullopt;
    if (!bytes) {
        log_warning(tt::LogOp, "KDA performance-model physical byte count overflowed; returning a zero estimate");
        return std::nullopt;
    }
    return KdaTensorTraffic{
        .buffer_address = tensor.buffer()->address(),
        .physical_bytes = *bytes,
        .is_dram = tensor.memory_config().is_dram(),
    };
}

KdaEstimate zero_estimate(std::size_t input_count, std::size_t output_count) {
    return KdaEstimate{
        .input_bytes = std::vector<uint64_t>(input_count, 0),
        .output_bytes = std::vector<uint64_t>(output_count, 0),
    };
}

KdaEstimate estimate(
    const KdaWork& work,
    std::span<const KdaTensorTraffic> inputs,
    std::span<const KdaTensorTraffic> outputs,
    uint64_t core_count,
    uint64_t clock_mhz,
    tt::tt_metal::MathFidelity math_fidelity) {
    KdaEstimate result = zero_estimate(inputs.size(), outputs.size());
    const auto factor = fidelity_factor(math_fidelity);
    if (core_count == 0 || clock_mhz == 0 || !factor) {
        log_warning(
            tt::LogOp,
            "KDA performance model received zero cores, zero clock, or invalid fidelity; returning a zero estimate");
        return result;
    }

    const auto cycle_numerator = checked_sum(
        {checked_product({Wide{work.dense_flops}, Wide{*factor}}),
         checked_product({Wide{32}, Wide{work.multiply_results}, Wide{*factor}}),
         checked_product({Wide{32}, Wide{work.add_results}}),
         checked_product({Wide{16}, Wide{work.reduction_input_elements}, Wide{*factor}})});
    const auto cycle_denominator = checked_product({Wide{MATRIX_FLOPS_PER_CORE_CYCLE}, Wide{core_count}});
    if (!cycle_numerator || !cycle_denominator) {
        log_warning(tt::LogOp, "KDA performance-model cycle arithmetic overflowed; returning a zero estimate");
        return result;
    }
    const Wide ideal_fpu_cycles = ceil_div(*cycle_numerator, *cycle_denominator);
    const auto fpu_time_numerator = checked_product({ideal_fpu_cycles, Wide{1000}});
    if (!fpu_time_numerator) {
        log_warning(tt::LogOp, "KDA performance-model time arithmetic overflowed; returning a zero estimate");
        return result;
    }
    const Wide ideal_fpu_ns = ceil_div(*fpu_time_numerator, clock_mhz);

    Wide mandatory_dram_bytes = 0;
    std::unordered_set<uint64_t> input_addresses;
    std::unordered_set<uint64_t> output_addresses;
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        const auto& tensor = inputs[index];
        if (tensor.is_dram) {
            if (tensor.physical_bytes > PROFILER_INT_MAX) {
                log_warning(tt::LogOp, "KDA performance-model input byte count does not fit profiler int");
                return zero_estimate(inputs.size(), outputs.size());
            }
            result.input_bytes[index] = tensor.physical_bytes;
            if (input_addresses.insert(tensor.buffer_address).second) {
                const auto total = checked_sum({mandatory_dram_bytes, Wide{tensor.physical_bytes}});
                if (!total) {
                    log_warning(tt::LogOp, "KDA performance-model DRAM input bytes overflowed");
                    return zero_estimate(inputs.size(), outputs.size());
                }
                mandatory_dram_bytes = *total;
            }
        }
    }
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        const auto& tensor = outputs[index];
        if (tensor.is_dram) {
            if (tensor.physical_bytes > PROFILER_INT_MAX) {
                log_warning(tt::LogOp, "KDA performance-model output byte count does not fit profiler int");
                return zero_estimate(inputs.size(), outputs.size());
            }
            result.output_bytes[index] = tensor.physical_bytes;
            if (output_addresses.insert(tensor.buffer_address).second) {
                const auto total = checked_sum({mandatory_dram_bytes, Wide{tensor.physical_bytes}});
                if (!total) {
                    log_warning(tt::LogOp, "KDA performance-model DRAM output bytes overflowed");
                    return zero_estimate(inputs.size(), outputs.size());
                }
                mandatory_dram_bytes = *total;
            }
        }
    }

    const Wide ideal_dram_ns = ceil_div(mandatory_dram_bytes, DRAM_BYTES_PER_NS);
    const Wide ideal_ns = std::max(ideal_fpu_ns, ideal_dram_ns);
    if (!fits_profiler_int(ideal_fpu_cycles) || !fits_profiler_int(ideal_fpu_ns) || !fits_profiler_int(ideal_dram_ns) ||
        !fits_profiler_int(ideal_ns) || mandatory_dram_bytes > U64_MAX) {
        log_warning(
            tt::LogOp, "KDA performance-model estimate does not fit profiler fields; returning a zero estimate");
        return zero_estimate(inputs.size(), outputs.size());
    }

    result.valid = true;
    result.ideal_fpu_cycles = static_cast<uint64_t>(ideal_fpu_cycles);
    result.ideal_fpu_ns = static_cast<uint64_t>(ideal_fpu_ns);
    result.mandatory_dram_bytes = static_cast<uint64_t>(mandatory_dram_bytes);
    result.ideal_dram_ns = static_cast<uint64_t>(ideal_dram_ns);
    result.ideal_ns = static_cast<uint64_t>(ideal_ns);
    result.omitted_sfpu_results = work.omitted_sfpu_results;
    return result;
}

}  // namespace ttnn::experimental::prim::kda_performance_model

// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <vector>
#include <cstdint>

namespace ttnn {
namespace operations::experimental::adaptive_pool {

// Structure for PyTorch's exact adaptive pooling regions
struct AdaptiveRegion {
    uint32_t output_h, output_w;              // Output position
    uint32_t start_h, end_h, start_w, end_w;  // Input region boundaries
    uint32_t kernel_h, kernel_w;              // Region size
};

// SAFE HYBRID PADDING+DILATION approach with PyTorch compatibility
enum class AdaptivePoolStrategy {
    PURE_UNIFORM,       // Already uniform - no adjustment needed
    PADDING_DOMINANT,   // Legacy padding approach for compatibility (default for edge cases)
    DILATION_DOMINANT,  // Conservative dilation for low variance cases
    COMBINED_OPTIMAL    // Balanced approach with compatibility checks
};

struct HybridAdaptiveConfig {
    AdaptivePoolStrategy strategy;

    // Padding configuration
    std::array<uint32_t, 4> padding;  // [pad_top, pad_bottom, pad_left, pad_right]

    // Dilation configuration
    std::array<uint32_t, 2> dilation;  // [dilation_h, dilation_w]

    // Resulting uniform parameters
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;

    // Analysis metrics
    uint32_t h_variance, w_variance;              // Kernel variance before adjustment
    uint32_t h_variance_after, w_variance_after;  // Kernel variance after adjustment
    double coverage_improvement_percent;          // Improvement over legacy approach (conservative)
    double memory_overhead_percent;               // Total memory cost

    bool is_beneficial() const {
        return coverage_improvement_percent > 0.0;  // Any improvement is considered beneficial
    }
};

// Legacy structure for backward compatibility
struct AdaptivePoolingParams {
    std::array<uint32_t, 2> kernel_size;
    std::array<uint32_t, 2> stride;
    std::array<uint32_t, 4> padding;  // [pad_top, pad_bottom, pad_left, pad_right]
    uint32_t padded_h;
    uint32_t padded_w;
    double memory_overhead_percent;
};

// Core utility functions
std::vector<AdaptiveRegion> calculate_pytorch_regions(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w);

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> analyze_adaptive_kernels(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w);

std::pair<uint32_t, uint32_t> calculate_kernel_variance(const std::vector<uint32_t>& kernels);

HybridAdaptiveConfig calculate_pattern_based_hybrid_config(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w);

AdaptivePoolingParams convert_hybrid_to_legacy(
    const HybridAdaptiveConfig& hybrid_config, uint32_t input_h, uint32_t input_w);

}  // namespace operations::experimental::adaptive_pool
}  // namespace ttnn

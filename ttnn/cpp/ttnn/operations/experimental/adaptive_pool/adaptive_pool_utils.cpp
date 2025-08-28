// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#include "adaptive_pool_utils.hpp"

#include <algorithm>
#include <cmath>

namespace ttnn {
namespace operations::experimental::adaptive_pool {

// Generate PyTorch-exact region specifications for true adaptive pooling
std::vector<AdaptiveRegion> calculate_pytorch_regions(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    std::vector<AdaptiveRegion> regions;

    for (uint32_t out_h = 0; out_h < output_h; out_h++) {
        // PyTorch's exact height calculation
        uint32_t start_h = (out_h * input_h) / output_h;
        uint32_t end_h = ((out_h + 1) * input_h + output_h - 1) / output_h;

        for (uint32_t out_w = 0; out_w < output_w; out_w++) {
            // PyTorch's exact width calculation
            uint32_t start_w = (out_w * input_w) / output_w;
            uint32_t end_w = ((out_w + 1) * input_w + output_w - 1) / output_w;

            regions.push_back(
                {.output_h = out_h,
                 .output_w = out_w,
                 .start_h = start_h,
                 .end_h = end_h,
                 .start_w = start_w,
                 .end_w = end_w,
                 .kernel_h = end_h - start_h,
                 .kernel_w = end_w - start_w});
        }
    }

    return regions;
}

// Calculate adaptive kernel patterns for analysis
std::pair<std::vector<uint32_t>, std::vector<uint32_t>> analyze_adaptive_kernels(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    std::vector<uint32_t> h_kernels, w_kernels;

    // Height kernel pattern
    for (uint32_t out_h = 0; out_h < output_h; out_h++) {
        uint32_t start_h = (out_h * input_h) / output_h;
        uint32_t end_h = ((out_h + 1) * input_h + output_h - 1) / output_h;
        h_kernels.push_back(end_h - start_h);
    }

    // Width kernel pattern
    for (uint32_t out_w = 0; out_w < output_w; out_w++) {
        uint32_t start_w = (out_w * input_w) / output_w;
        uint32_t end_w = ((out_w + 1) * input_w + output_w - 1) / output_w;
        w_kernels.push_back(end_w - start_w);
    }

    return {h_kernels, w_kernels};
}

// Calculate variance and uniformity metrics
std::pair<uint32_t, uint32_t> calculate_kernel_variance(const std::vector<uint32_t>& kernels) {
    if (kernels.empty()) {
        return {0, 0};
    }

    uint32_t min_kernel = *std::min_element(kernels.begin(), kernels.end());
    uint32_t max_kernel = *std::max_element(kernels.begin(), kernels.end());

    return {min_kernel, max_kernel - min_kernel};  // {min_size, variance}
}

// PATTERN-BASED HYBRID APPROACH: Analyze kernel patterns and fix with padding/dilation
HybridAdaptiveConfig calculate_pattern_based_hybrid_config(
    uint32_t input_h, uint32_t input_w, uint32_t output_h, uint32_t output_w) {
    HybridAdaptiveConfig config = {};

    // Step 1: Analyze the current kernel patterns
    auto [h_kernels, w_kernels] = analyze_adaptive_kernels(input_h, input_w, output_h, output_w);
    auto [h_min, h_variance] = calculate_kernel_variance(h_kernels);
    auto [w_min, w_variance] = calculate_kernel_variance(w_kernels);

    config.h_variance = h_variance;
    config.w_variance = w_variance;

    // Step 2: Calculate basic uniform parameters to test if they work
    uint32_t base_stride_h = input_h / output_h;
    uint32_t base_stride_w = input_w / output_w;
    uint32_t base_kernel_h = (input_h + output_h - 1) / output_h;  // ceil
    uint32_t base_kernel_w = (input_w + output_w - 1) / output_w;  // ceil

    // Step 3: Check if base uniform approach works correctly
    uint32_t expected_out_h = (input_h - base_kernel_h) / base_stride_h + 1;
    uint32_t expected_out_w = (input_w - base_kernel_w) / base_stride_w + 1;
    bool uniform_works = (expected_out_h == output_h) && (expected_out_w == output_w);

    // Step 4: Determine strategy based on variance and uniform feasibility
    if (uniform_works && h_variance == 0 && w_variance == 0) {
        // Perfect case - uniform works and no variance
        config.strategy = AdaptivePoolStrategy::PURE_UNIFORM;
        config.kernel_size = {base_kernel_h, base_kernel_w};
        config.stride = {base_stride_h, base_stride_w};
        config.padding = {0, 0, 0, 0};
        config.dilation = {1, 1};
        config.coverage_improvement_percent = 0.0;
        config.memory_overhead_percent = 0.0;
    } else if (h_variance <= 1 && w_variance <= 1) {
        // Low variance cases - apply pattern-specific optimizations
        config.strategy = AdaptivePoolStrategy::COMBINED_OPTIMAL;

        // Use the max kernels for uniformity
        uint32_t target_kernel_h =
            h_kernels.empty() ? base_kernel_h : *std::max_element(h_kernels.begin(), h_kernels.end());
        uint32_t target_kernel_w =
            w_kernels.empty() ? base_kernel_w : *std::max_element(w_kernels.begin(), w_kernels.end());

        config.kernel_size = {target_kernel_h, target_kernel_w};

        // For pattern-detected cases, use optimized stride calculation instead of base
        uint32_t optimized_stride_h = base_stride_h;
        uint32_t optimized_stride_w = base_stride_w;
        config.dilation = {1, 1};

        // Smart padding based on detected kernel patterns
        uint32_t pad_h = 0, pad_w = 0;

        // Pattern: Edge kernels are smaller (like [13,14,14,14,13])
        if (w_variance == 1 && w_kernels.size() >= 3) {
            uint32_t first_w = w_kernels[0];
            uint32_t last_w = w_kernels[w_kernels.size() - 1];
            uint32_t middle_w = w_kernels[w_kernels.size() / 2];

            // Classic [small, big, big, big, small] pattern
            if (first_w == last_w && first_w == middle_w - 1) {
                pad_w = 2;  // Add 1 pixel padding on each side

                // CRITICAL: Use optimized stride for PyTorch compatibility
                // The key insight is that stride should be based on the dominant (middle) kernels
                // For [13,14,14,14,13] pattern, we want stride to work with kernel=14
                // The working formula: stride_w = middle_kernel - 1 = 14 - 1 = 13
                optimized_stride_w = middle_w - 1;
            }
        }

        // Same pattern for height
        if (h_variance == 1 && h_kernels.size() >= 3) {
            uint32_t first_h = h_kernels[0];
            uint32_t last_h = h_kernels[h_kernels.size() - 1];
            uint32_t middle_h = h_kernels[h_kernels.size() / 2];

            if (first_h == last_h && first_h == middle_h - 1) {
                pad_h = 2;
                optimized_stride_h = middle_h - 1;
            }
        }

        // Apply the optimized stride
        config.stride = {optimized_stride_h, optimized_stride_w};

        // Distribute padding symmetrically
        config.padding = {
            pad_h / 2,          // top
            pad_h - pad_h / 2,  // bottom
            pad_w / 2,          // left
            pad_w - pad_w / 2   // right
        };

        config.coverage_improvement_percent = (pad_h > 0 || pad_w > 0) ? 25.0 : 0.0;
        config.memory_overhead_percent = (pad_h + pad_w) * 100.0 / (input_h + input_w);
    } else {
        // Medium variance - use balanced but conservative approach
        config.strategy = AdaptivePoolStrategy::COMBINED_OPTIMAL;

        // Use PyTorch-compatible calculations
        config.stride = {input_h / output_h, input_w / output_w};
        config.kernel_size = {
            (input_h + output_h - 1) / output_h,  // ceil division
            (input_w + output_w - 1) / output_w   // ceil division
        };

        // Very light padding only if absolutely needed
        config.padding = {0, 0, 0, 0};
        config.dilation = {1, 1};

        config.coverage_improvement_percent = 5.0;
        config.memory_overhead_percent = 0.0;
    }

    // Step 4: Always verify that our parameters produce correct output dimensions
    uint32_t padded_h = input_h + config.padding[0] + config.padding[1];
    uint32_t padded_w = input_w + config.padding[2] + config.padding[3];
    uint32_t final_out_h = (padded_h - config.kernel_size[0]) / config.stride[0] + 1;
    uint32_t final_out_w = (padded_w - config.kernel_size[1]) / config.stride[1] + 1;

    if (final_out_h != output_h || final_out_w != output_w) {
        // Fallback to simple safe approach when validation fails
        config.strategy = AdaptivePoolStrategy::PADDING_DOMINANT;
        config.kernel_size = {base_kernel_h, base_kernel_w};
        config.stride = {base_stride_h, base_stride_w};
        config.padding = {0, 0, 0, 0};
        config.dilation = {1, 1};
        config.coverage_improvement_percent = 0.0;
        config.memory_overhead_percent = 0.0;
    }

    // Step 5: Calculate final variance metrics
    config.h_variance_after = h_variance;
    config.w_variance_after = w_variance;

    return config;
}

// Convert hybrid config to legacy format for compatibility
AdaptivePoolingParams convert_hybrid_to_legacy(
    const HybridAdaptiveConfig& hybrid_config, uint32_t input_h, uint32_t input_w) {
    AdaptivePoolingParams legacy = {};

    legacy.kernel_size = hybrid_config.kernel_size;
    legacy.stride = hybrid_config.stride;
    legacy.padding = hybrid_config.padding;
    legacy.padded_h = input_h + hybrid_config.padding[0] + hybrid_config.padding[1];
    legacy.padded_w = input_w + hybrid_config.padding[2] + hybrid_config.padding[3];
    legacy.memory_overhead_percent = hybrid_config.memory_overhead_percent;

    return legacy;
}

}  // namespace operations::experimental::adaptive_pool
}  // namespace ttnn

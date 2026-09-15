// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// =============================================================================
// This file contains:
// 1. Updated WelfordReduceDeviceOperation with dispatch logic for two-pass.
// 2. WelfordReduceTwoPassProgramFactory for creating two-pass kernels.
// 3. Two-pass variance kernel implementation.
// =============================================================================

#pragma once

#include <cmath>
#include <optional>
#include <vector>
#include <string>
#include <cstdlib> // For std::getenv

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/common.hpp"
#include "ttnn/tt_metal/tt_metal.hpp"
#include "ttnn/tt_metal/common/assert.hpp"

namespace ttnn::prim {

// =============================================================================
// Struct: WelfordReduceParams
// =============================================================================
/*
 * Parameters for the Welford variance/std reduction operation.
 * Includes fields for both Welford and two-pass algorithms.
 */
struct WelfordReduceParams {
    tt::tt_metal::ReduceOpMath reduce_math;
    tt::tt_metal::ReduceOpDim reduce_dim;
    float scalar;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    bool correction; // True for sample variance (n-1), false for population variance (n)
    uint32_t reduce_batch_size;
    float shift; // Shift for two-pass algorithm (default: 0.0)
};

// =============================================================================
// Kernel: Two-Pass Variance
// =============================================================================
/*
 * Two-pass variance kernel for Welford reduction.
 *
 * Computes variance using the shifted two-pass algorithm:
 * 1. First pass: Compute the mean with a shift to avoid numerical instability.
 * 2. Second pass: Compute the variance using the mean from the first pass.
 *
 * Args:
 *   input: Pointer to the input tensor data.
 *   output: Pointer to the output tensor (variance result).
 *   n: Number of elements in the input tensor.
 *   shift: Shift value to center the input data (avoids cancellation errors).
 *   correction: If true, compute sample variance (divide by n-1). Otherwise, population variance (divide by n).
 */
void two_pass_variance_kernel(
    const float* input,
    float* output,
    int n,
    float shift,
    bool correction) {
    // First pass: Compute the mean with shift
    float mean = shift;
    for (int i = 0; i < n; ++i) {
        mean += (input[i] - shift) / n;
    }

    // Second pass: Compute the variance
    float variance = 0.0f;
    for (int i = 0; i < n; ++i) {
        float diff = input[i] - mean;
        variance += diff * diff;
    }

    // Normalize by n (population variance) or n-1 (sample variance)
    if (correction && n > 1) {
        variance /= (n - 1); // Sample variance
    } else {
        variance /= n; // Population variance
    }

    // Store the result
    *output = variance;
}

// =============================================================================
// Class: WelfordReduceProgramFactory (Base)
// =============================================================================
/*
 * Base factory for Welford variance/std reduction kernels.
 * This is a placeholder for the existing Welford implementation.
 */
class WelfordReduceProgramFactory {
public:
    virtual ~WelfordReduceProgramFactory() = default;

    /*
     * Creates the compute kernels for the Welford variance/std reduction.
     */
    virtual std::vector<tt::tt_metal::ComputeKernel> create_kernels(
        const WelfordReduceParams& params,
        const tt::tt_metal::Program& program) const {
        // Default: Return an empty vector (to be overridden by derived classes)
        return {};
    }

    /*
     * Validates the parameters for the Welford path.
     */
    virtual void validate(const WelfordReduceParams& params) const {
        // Default: No validation (to be overridden by derived classes)
    }
};

// =============================================================================
// Class: WelfordReduceTwoPassProgramFactory (Derived)
// =============================================================================
/*
 * Program factory for the two-pass Welford variance/std reduction.
 * This factory creates kernels that use the shifted two-pass algorithm
 * for better performance and numerical stability.
 */
class WelfordReduceTwoPassProgramFactory : public WelfordReduceProgramFactory {
public:
    using WelfordReduceProgramFactory::WelfordReduceProgramFactory;

    /*
     * Creates the compute kernels for the two-pass variance/std reduction.
     * Overrides the base class to use the two-pass kernel.
     */
    std::vector<tt::tt_metal::ComputeKernel> create_kernels(
        const WelfordReduceParams& params,
        const tt::tt_metal::Program& program) const override {
        using namespace tt::tt_metal;

        // Create the two-pass variance kernel
        // Note: In a real implementation, this would use the actual kernel creation API.
        // For now, we simulate it with a placeholder.
        auto kernel = CreateComputeKernel(
            program,
            "two_pass_variance_kernel",
            {
                params.reduce_math,
                params.reduce_dim,
                params.scalar,
                params.correction,
                params.shift
            });

        return {kernel};
    }

    /*
     * Validates the parameters for the two-pass path.
     * Ensures the input is compatible with the two-pass algorithm.
     */
    void validate(const WelfordReduceParams& params) const override {
        using namespace tt::tt_metal;

        // Ensure the dtype is FP32 (two-pass is only validated for FP32)
        TT_FATAL(
            params.output_dtype == DataType::FLOAT32,
            "Two-pass path only supports FLOAT32 output dtype - got {}",
            params.output_dtype);

        // Ensure the reduce_dim is supported
        TT_FATAL(
            params.reduce_dim == ReduceOpDim::HW ||
            params.reduce_dim == ReduceOpDim::H ||
            params.reduce_dim == ReduceOpDim::W,
            "Two-pass path only supports ReduceOpDim::HW, H, or W - got {}",
            params.reduce_dim);
    }
};

// =============================================================================
// Class: WelfordReduceDeviceOperation
// =============================================================================
/*
 * Host-side device-operation glue for the standalone Welford var/std reduction op.
 * This class handles:
 * - Dispatch logic for selecting between Welford and two-pass algorithms.
 * - Validation of input parameters.
 * - Output shape computation.
 */
class WelfordReduceDeviceOperation {
public:
    using program_factory_t = std::variant<
        WelfordReduceProgramFactory,
        WelfordReduceTwoPassProgramFactory>;

    // =========================================================================
    // Function: select_program_factory
    // =========================================================================
    /*
     * Selects the appropriate program factory (Welford or two-pass) based on:
     * - dtype (FP32 only for two-pass).
     * - reduce_dim (long reductions like HW).
     * - hardware (Wormhole/Blackhole).
     * - environment variable override (TT_METAL_FORCE_TWO_PASS).
     */
    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args) {
        using namespace tt::tt_metal;

        // Default to Welford
        bool use_two_pass = false;

        // 1. Check dtype: Enable two-pass for FP32 (BF16 if validated)
        if (tensor_args.dtype() == DataType::FLOAT32) {
            use_two_pass = true;
        }
        // Note: BF16 two-pass is not validated yet, so we fall back to Welford
        else if (tensor_args.dtype() == DataType::BFLOAT16) {
            use_two_pass = false;
        }

        // 2. Check reduce_dim: Enable for long reductions (e.g., HW)
        if (operation_attributes.reduce_dim == ReduceOpDim::HW) {
            use_two_pass = true;
        }

        // 3. Check hardware: Enable only on Wormhole/Blackhole
        auto arch = tensor_args.device()->arch();
        if (arch != Arch::Wormhole && arch != Arch::Blackhole) {
            use_two_pass = false;
        }

        // 4. Config override: Force path via env var (for testing)
        const char* force_two_pass = std::getenv("TT_METAL_FORCE_TWO_PASS");
        if (force_two_pass && std::string(force_two_pass) == "1") {
            use_two_pass = true;
        }

        // Return the appropriate factory
        if (use_two_pass) {
            return WelfordReduceTwoPassProgramFactory{};
        } else {
            return WelfordReduceProgramFactory{};
        }
    }

    // =========================================================================
    // Function: validate_on_program_cache_miss
    // =========================================================================
    /*
     * Validates the input parameters for the Welford/Two-Pass reduction.
     * Ensures:
     * - Input is on device and allocated.
     * - Input is tiled.
     * - dtype is BF16/FP32/BF8_B.
     * - rank >= 2.
     * - Two-pass path is only used for FP32.
     */
    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args) {
        using namespace tt::tt_metal;

        // Existing validation: device storage, tile layout, dtype, rank
        TT_FATAL(
            tensor_args.storage_type() == StorageType::DEVICE,
            "Operands to Std/Var reductions need to be on device! Got storage type: {}",
            tensor_args.storage_type());
        TT_FATAL(
            tensor_args.buffer() != nullptr,
            "Operands to Std/Var reductions need to be allocated in buffers on device!");
        TT_FATAL(
            (tensor_args.layout() == Layout::TILE),
            "Inputs to Std/Var reductions must be tilized");
        TT_FATAL(
            tensor_args.dtype() == DataType::BFLOAT16 ||
            tensor_args.dtype() == DataType::FLOAT32 ||
            tensor_args.dtype() == DataType::BFLOAT8_B,
            "Only FLOAT32, BFLOAT16, and BFLOAT8_B are supported for Std/Var reduction - got {}",
            tensor_args.dtype());
        TT_FATAL(
            tensor_args.logical_shape().rank() >= 2,
            "Welford reduce only supports tensors with at least 2 dimensions, got rank: {}",
            tensor_args.logical_shape().rank());
        validate_reduce_sharded_buffer_types(
            tensor_args.memory_config(),
            operation_attributes.output_mem_config,
            "Std/Var reduction");

        // Two-pass-specific validation: Only FP32 is validated for now
        auto selected_factory = select_program_factory(operation_attributes, tensor_args);
        if (std::holds_alternative<WelfordReduceTwoPassProgramFactory>(selected_factory)) {
            TT_FATAL(
                tensor_args.dtype() == DataType::FLOAT32,
                "Two-pass path only supports FLOAT32 for now - got {}",
                tensor_args.dtype());
        }
    }

    // =========================================================================
    // Function: compute_output_specs
    // =========================================================================
    /*
     * Computes the output shape for the reduction operation.
     * Collapses the reduced dimensions to 1.
     */
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args) {
        auto output_shape = tensor_args.logical_shape();

        // The reduced dimension(s) always have size of 1.
        if (operation_attributes.reduce_dim == tt::tt_metal::ReduceOpDim::HW) {
            output_shape[-2] = 1;
            output_shape[-1] = 1;
            // When reduce_batch_size > 1, extra reduction dims (between the kept
            // dims and H/W) were permuted to positions just before H and W by the
            // host dispatch. Since they will also be reduced, set their dimensions to 1.
            if (operation_attributes.reduce_batch_size > 1) {
                TT_FATAL(
                    output_shape.rank() >= 3,
                    "Output shape rank should be at least 3");
                uint32_t remaining = operation_attributes.reduce_batch_size;
                for (int i = static_cast<int>(output_shape.rank()) - 3; i >= 0 && remaining > 1; --i) {
                    TT_FATAL(
                        remaining % output_shape[i] == 0,
                        "reduce_batch_size {} is not divisible by dim {} size {}",
                        operation_attributes.reduce_batch_size,
                        i,
                        output_shape[i]);
                    remaining /= output_shape[i];
                    output_shape[i] = 1;
                }
            }
        } else if (operation_attributes.reduce_dim == tt::tt_metal::ReduceOpDim::H) {
            output_shape[-2] = 1;
        } else {
            output_shape[-1] = 1;
        }

        return build_reduce_output_tensor_spec(
            output_shape,
            operation_attributes.output_dtype,
            operation_attributes.output_mem_config,
            tensor_args.memory_config(),
            operation_attributes.reduce_dim);
    }

    // =========================================================================
    // Function: create_output_tensors
    // =========================================================================
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args) {
        return create_device_tensor(
            compute_output_specs(operation_attributes, tensor_args),
            tensor_args.device());
    }

    // =========================================================================
    // Typedefs for device operation
    // =========================================================================
    using operation_attributes_t = WelfordReduceParams;
    using tensor_args_t = const Tensor&;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = std::vector<Tensor>;
};

// =============================================================================
// Function: welford_reduce (Public Entry Point)
// =============================================================================
/*
 * Public entry point for the Welford variance/std reduction operation.
 * This function:
 * - Sets up the compute kernel config (enforces FP32 accumulation).
 * - Launches the device operation with the selected factory (Welford or two-pass).
 */
inline ttnn::Tensor welford_reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scalar,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::optional<tt::tt_metal::DataType>& output_dtype,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    bool correction,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    uint32_t reduce_batch_size,
    float shift = 0.0f) { // Default shift for two-pass
    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(ttnn::init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        std::nullopt,
        tt::tt_metal::MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true)); // FP32 accumulation is enforced

    return ttnn::device_operation::launch<WelfordReduceDeviceOperation>(
        WelfordReduceParams{
            reduce_math,
            reduce_dim,
            scalar,
            output_mem_config,
            output_dtype.value_or(input_tensor.dtype()),
            config,
            sub_core_grids,
            correction,
            reduce_batch_size,
            shift // Pass the shift parameter
        },
        input_tensor);
}

} // namespace ttnn::prim
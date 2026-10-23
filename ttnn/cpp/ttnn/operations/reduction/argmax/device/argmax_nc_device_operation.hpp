// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

// Parameters for the register-based argmax over a non-HW (outer / "NC") dim.
// Operates on TILE-layout inputs of bf16 or fp32 and produces a TILE-layout
// uint32 tensor with the reduced dim replaced by a single tile (keepdim=true
// internally, the caller is responsible for any post-processing).
struct ArgMaxNCParams {
    int32_t dim{};
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<CoreRangeSet> sub_core_grids;
};

struct ArgMaxNCInputs {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

struct ArgMaxNCDeviceOperation {
    using operation_attributes_t = ArgMaxNCParams;
    using tensor_args_t = ArgMaxNCInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // Addresses are Buffer bindings; every other runtime arg is fixed by the hashed specs.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Facade: returns a TILE-layout UINT32 tensor with the reduced dim's padded
// extent collapsed to a single tile (i.e. the output has the same padded shape
// as the input but `tile_height`/`tile_width` in the reduced dim position --
// typically one tile). Post-processing (slicing to logical size, layout
// conversion, etc.) is the caller's responsibility.
Tensor argmax_nc(
    const Tensor& input,
    int32_t dim,
    const std::optional<Tensor>& preallocated_output,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids = std::nullopt);

}  // namespace ttnn::prim

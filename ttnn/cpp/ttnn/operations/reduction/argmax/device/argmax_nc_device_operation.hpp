// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"

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

struct ArgMaxNCProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        uint32_t num_cores_to_be_used{};
        uint32_t num_cores_x{};
        std::vector<tt::tt_metal::CoreCoord> ordered_cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ArgMaxNCParams& operation_attributes, const ArgMaxNCInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ArgMaxNCParams& operation_attributes,
        const ArgMaxNCInputs& tensor_args,
        Tensor& tensor_return_value);
};

struct ArgMaxNCDeviceOperation {
    using operation_attributes_t = ArgMaxNCParams;
    using tensor_args_t = ArgMaxNCInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<ArgMaxNCProgramFactory>;

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

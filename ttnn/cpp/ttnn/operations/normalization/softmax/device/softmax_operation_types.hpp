// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/core_coord.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include <cstdint>
#include <optional>

namespace ttnn {
// Softmax program configuration structs
struct SoftmaxDefaultProgramConfig {};
struct SoftmaxShardedMultiCoreProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t subblock_w{};
    std::size_t block_h{};
    std::size_t block_w{};
};

using SoftmaxProgramConfig = std::variant<SoftmaxDefaultProgramConfig, SoftmaxShardedMultiCoreProgramConfig>;

// Softmax operation type
enum class SoftmaxOperationType : uint8_t {
    Softmax = 0,
    ScaleMaskSoftmax = 1,
    SoftmaxInPlace = 2,
    ScaleMaskSoftmaxInPlace = 3,
    ScaleCausalMaskHWSoftmaxInPlace = 4
};
}  // namespace ttnn

// Softmax operation structs
namespace ttnn::prim {

// Kernel path constants
inline constexpr const char* SOFTMAX_KERNEL_PATH_GENERAL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels";
inline constexpr const char* SOFTMAX_KERNEL_PATH_ATTENTION =
    "ttnn/cpp/ttnn/operations/normalization/softmax/device/kernels/attention";
struct SoftmaxParams {
    const SoftmaxOperationType softmax_type;
    const int8_t dim;
    const std::optional<float> scale;
    const bool inplace;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const SoftmaxProgramConfig program_config;
    const bool is_causal_mask;
    const DeviceComputeKernelConfig compute_kernel_config;
    const bool is_scale_causal_mask_hw_dims_softmax;
    const bool numeric_stable;
};

struct SoftmaxInputs {
    const Tensor& input_tensor;
    const std::optional<const Tensor> mask;
};

}  // namespace ttnn::prim

// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_softmax.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_softmax {

#define DEFINE_MOREH_SOFT_OP_INVOKE(name)                                                          \
    Tensor name::invoke(                                                                           \
        const Tensor& input_tensor,                                                                \
        uint32_t dim,                                                                              \
        const std::optional<Tensor>& output_tensor,                                                \
        const MorehSoftmaxOp op,                                                                   \
        const MorehSoftmaxOpParallelizationStrategy strategy,                                      \
        const std::optional<MemoryConfig>& memory_config,                                          \
        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {                   \
        return ttnn::prim::moreh_softmax(                                                          \
            input_tensor, dim, output_tensor, op, strategy, memory_config, compute_kernel_config); \
    }

DEFINE_MOREH_SOFT_OP_INVOKE(MorehSoftmax);
DEFINE_MOREH_SOFT_OP_INVOKE(MorehSoftmin);
DEFINE_MOREH_SOFT_OP_INVOKE(MorehLogSoftmax);
#undef DEFINE_MOREH_SOFT_OP_INVOKE

}  // namespace ttnn::operations::moreh::moreh_softmax

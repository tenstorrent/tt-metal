// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/metal_v2_artifacts.hpp>
#include <cstdint>

namespace ttnn::operations::moreh::moreh_softmax {

enum class MorehSoftmaxOpParallelizationStrategy {
    NONE,
    SMALL_W,
    SMALL_H,
    LARGE_W,
    LARGE_H,
    LARGE_C,
};

enum class MorehSoftmaxOp {
    SOFTMAX,
    SOFTMIN,
    LOGSOFTMAX,
};

bool is_moreh_softmax_w_small_available(const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config);
bool is_moreh_softmax_h_small_available(const Tensor& tensor, const DeviceComputeKernelConfig& compute_kernel_config);

struct MorehSoftmaxOperation {
    struct operation_attributes_t {
        std::uint32_t dim{};
        const MorehSoftmaxOp op;
        const MorehSoftmaxOpParallelizationStrategy strategy;
        const MemoryConfig memory_config;
        const DeviceComputeKernelConfig compute_kernel_config;
    };

    struct tensor_args_t {
        const Tensor& input;
        const std::optional<Tensor>& output;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

// All five factories ported to Metal 2.0 (MetalV2FactoryConcept); each returns ProgramArtifacts
// from create_program_artifacts. See device/softmax_*/softmax_*.cpp.
// NOTE (w_large): the fp32_dest_acc_en path must be built at -O3 (the w_large factory sets the compute
// KernelSpec opt_level to O3, matching legacy). At -O2 GCC fails to fold the LLK addrmod SETC16 inline-asm
// immediate in the larger fp32 TU and JIT aborts with "impossible constraint in 'asm'"; at O3 no workaround
// is needed. See moreh_softmax_w_large.cpp's top-of-file note.
#define DEFINE_SOFTMAX_FACTORY(factory_name)                                      \
    struct factory_name {                                                         \
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts( \
            const operation_attributes_t& operation_attributes,                   \
            const tensor_args_t& tensor_args,                                     \
            tensor_return_value_t& output);                                       \
    };

    DEFINE_SOFTMAX_FACTORY(MorehSoftmaxCLargeFactory)
    DEFINE_SOFTMAX_FACTORY(MorehSoftmaxHLargeFactory)
    DEFINE_SOFTMAX_FACTORY(MorehSoftmaxHSmallFactory)
    DEFINE_SOFTMAX_FACTORY(MorehSoftmaxWLargeFactory)
    DEFINE_SOFTMAX_FACTORY(MorehSoftmaxWSmallFactory)
#undef DEFINE_SOFTMAX_FACTORY

    using program_factory_t = std::variant<
        MorehSoftmaxCLargeFactory,
        MorehSoftmaxHLargeFactory,
        MorehSoftmaxWLargeFactory,
        MorehSoftmaxHSmallFactory,
        MorehSoftmaxWSmallFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_inputs(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static MorehSoftmaxOpParallelizationStrategy get_parallelization_strategy(
        const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::moreh::moreh_softmax

namespace ttnn::prim {
ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOperation::tensor_return_value_t moreh_softmax(
    const Tensor& input_tensor,
    std::uint32_t dim,
    const std::optional<Tensor>& output_tensor,
    ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOp op,
    ttnn::operations::moreh::moreh_softmax::MorehSoftmaxOpParallelizationStrategy strategy,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config);
}

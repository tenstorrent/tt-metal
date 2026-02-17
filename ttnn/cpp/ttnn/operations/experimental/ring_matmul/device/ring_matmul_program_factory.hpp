// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RingMatmulProgramFactory {
    struct shared_variables_t {
        std::vector<tt::tt_metal::KernelHandle> kernel_ids;  // [in0_reader, in1_reader, compute]
        std::vector<tt::tt_metal::CoreCoord> cores;
        uint32_t num_cores{};
        bool use_hop_cores{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RingMatmulParams& operation_attributes,
        const RingMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RingMatmulParams& operation_attributes,
        const RingMatmulInputs& tensor_args,
        std::vector<Tensor>& tensor_return_value);
};

// Helper function to create ring matmul program
RingMatmulProgramFactory::shared_variables_t ring_matmul_create_program(
    tt::tt_metal::Program& program,
    const Tensor& a,
    const std::vector<Tensor>& b_tensors,
    const std::optional<const Tensor>& bias,
    const std::vector<Tensor>& output_tensors,
    const RingMatmulConfig& config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<operations::unary::UnaryWithParam> fused_activation,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    bool untilize_out,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb);

}  // namespace ttnn::experimental::prim

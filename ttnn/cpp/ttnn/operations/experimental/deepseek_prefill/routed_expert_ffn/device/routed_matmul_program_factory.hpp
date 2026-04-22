// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "routed_matmul_types.hpp"

#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device {

// Forked 2D-mcast matmul program factory. Same layout/config as matmul's 2D mcast
// factory; the reader and compute kernels are replaced with forked variants that
// read a scalar from the max_expert_iter tensor and early-return when
// curr_expert_iter > max_expert_iter. curr_expert_iter is passed as a per-kernel runtime arg so the
// program can be cached across iterations.
struct RoutedMatmulMcast2DProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle mm_kernel_in0_sender_id{};
        std::vector<tt::tt_metal::CoreCoord> in0_sender_interleaved_cores;
        tt::tt_metal::KernelHandle mm_kernel_in1_sender_writer_id{};
        std::vector<tt::tt_metal::CoreCoord> in1_sender_cores;
        tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_id{};
        std::vector<tt::tt_metal::CoreCoord> in1_receiver_cores;
        tt::tt_metal::KernelHandle mm_kernel_in1_receiver_writer_other_noc_setup_id{};
        std::vector<tt::tt_metal::CoreCoord> in1_receiver_other_cores;
        tt::tt_metal::CBHandle cb_src2{};
        tt::tt_metal::CBHandle cb_output{};
        uint32_t num_cores_with_work_r{};
        uint32_t num_cores_with_work_c{};
        uint32_t start_core_x{};
        uint32_t start_core_y{};
        bool transpose_mcast{};
        std::vector<tt::tt_metal::CoreCoord> cores;
        // Guard mechanism
        tt::tt_metal::CBHandle cb_guard{};
        bool output_is_sharded{false};
        tt::tt_metal::KernelHandle mm_compute_kernel_id{};
        tt::tt_metal::KernelHandle mm_kernel_in0_receiver_id{};
        tt::tt_metal::KernelHandle mm_kernel_in0_receiver_other_noc_setup_id{};
        std::vector<tt::tt_metal::CoreCoord> in0_receiver_interleaved_cores;
        std::vector<tt::tt_metal::CoreCoord> in0_receiver_other_cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const RoutedMatmulParams& operation_attributes,
        const RoutedMatmulInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const RoutedMatmulParams& operation_attributes,
        const RoutedMatmulInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::routed_expert_ffn::device

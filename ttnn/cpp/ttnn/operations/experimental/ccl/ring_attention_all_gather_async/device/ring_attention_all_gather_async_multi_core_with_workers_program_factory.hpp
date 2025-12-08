// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <vector>

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async::program {

struct RingAttentionAllGatherAsyncMultiCoreWithWorkersSharedVariables {
    tt::tt_metal::KernelHandle worker_sender_reader_forward_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_forward_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_reader_backward_kernel_id{};
    tt::tt_metal::KernelHandle worker_sender_writer_backward_kernel_id{};
    std::vector<CoreCoord> sender_worker_cores;
    uint32_t num_inputs{};
    uint32_t reader_sender_rt_offset{};
    uint32_t writer_sender_rt_offset{};
    uint32_t num_links{};
};

struct RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory {
    using shared_variables_t = RingAttentionAllGatherAsyncMultiCoreWithWorkersSharedVariables;

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    using operation_attributes_t = ring_attention_all_gather_async::operation_attributes_t;

    using tensor_args_t = ring_attention_all_gather_async::tensor_args_t;

    using tensor_return_value_t = ring_attention_all_gather_async::tensor_return_value_t;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async::program

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async.hpp"
#include <utility>
#include "ttnn/operations/experimental/ccl/all_to_all_async/device/all_to_all_async_op.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::experimental::ccl {

ttnn::Tensor ExecuteAllToAllAsync::invoke(
    const ttnn::Tensor& input_tensor,
    ttnn::Tensor& persistent_intermediate_buffer,
    ttnn::Tensor& persistent_output_buffer,
    const int32_t in_dim,
    const int32_t out_dim,
    const GlobalSemaphore& multi_device_global_semaphore,
    const uint32_t num_links,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    const ttnn::ccl::Topology topology,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id) {
    bool composite_all_to_all_case =
        composite_common::use_composite_all_to_all(input_tensor, in_dim, out_dim, memory_config);
    if (composite_all_to_all_case) {
        // Ugly code: the native implementation forces the user to pre-allocate the output buffer,
        // but the composite implementation is unable to reuse it. So overwrite persistent_output_buffer
        // to point to the real output buffer internally created by the composite implementation.
        persistent_output_buffer = composite_common::composite_all_to_all(
            input_tensor, in_dim, out_dim, num_links, memory_config, subdevice_id);
        return persistent_output_buffer;
    } else {
        return ttnn::operations::experimental::ccl::all_to_all_async(
            input_tensor,
            persistent_intermediate_buffer,
            persistent_output_buffer,
            in_dim,
            out_dim,
            multi_device_global_semaphore,
            num_links,
            memory_config,
            topology,
            subdevice_id);
    }
}

}  // namespace ttnn::operations::experimental::ccl

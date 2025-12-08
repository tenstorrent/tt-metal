// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/global_semaphore.hpp"
#include <optional>
#include <vector>

namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async {

struct operation_attributes_t {
    IDevice* target_device;
    std::optional<IDevice*> forward_device;
    std::optional<IDevice*> backward_device;
    uint32_t dim;
    uint32_t num_links;
    uint32_t ring_size;
    uint32_t ring_index;
    ccl::Topology topology;
    const std::vector<GlobalSemaphore>& semaphore;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler;
    CoreCoord core_grid_offset;
};

struct tensor_args_t {
    const std::vector<Tensor>& input_tensor;
};

using tensor_return_value_t = std::vector<Tensor>&;

using spec_return_value_t = std::vector<TensorSpec>;

}  // namespace ttnn::operations::experimental::ccl::ring_attention_all_gather_async

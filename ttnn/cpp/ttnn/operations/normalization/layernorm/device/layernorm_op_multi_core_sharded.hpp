// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/core.hpp"

#include "ttnn/operations/normalization/layernorm/device/layernorm_types.hpp"

namespace ttnn::operations::normalization {

struct LayerNormShardedOverrideRuntimeArgumentsCapture {
    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    tt::tt_metal::KernelHandle writer_mcast_sender_kernels_id = {};
    tt::tt_metal::KernelHandle writer_mcast_receiver_kernels_id = {};
    uint32_t num_none_all_to_all_workers = 0;
    bool is_pre_all_gather = false;
    tt::tt_metal::CBHandle cb_in0{};
    tt::tt_metal::CBHandle cb_in1{};
    tt::tt_metal::CBHandle cb_stats{};
    tt::tt_metal::CBHandle cb_add_out{};
    tt::tt_metal::CBHandle cb_output{};
    std::vector<tt::tt_metal::CoreCoord> cores;
};

tt::tt_metal::operation::ProgramWithCallbacks layernorm_multi_core_sharded(
    const Tensor& a,
    const std::optional<const Tensor>& b,
    const std::optional<const Tensor>& gamma,
    const std::optional<const Tensor>& beta,
    const std::optional<const Tensor>& stats,
    Tensor& output,
    LayerNormType norm_type,
    DistributedLayerNormStage distributed_norm_stage,
    float eps,
    CoreCoord compute_grid_size,
    uint32_t subblock_wt,
    uint32_t block_ht,
    uint32_t block_wt,
    bool legacy_reduction,
    bool legacy_rsqrt,
    bool use_welford,
    DeviceComputeKernelConfig compute_kernel_config);

void update_layernorm_multi_core_sharded_args(
    const LayerNormShardedOverrideRuntimeArgumentsCapture& capture,
    const void* operation,
    Program& program,
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    const std::vector<Tensor>& output_tensors);

}  // namespace ttnn::operations::normalization

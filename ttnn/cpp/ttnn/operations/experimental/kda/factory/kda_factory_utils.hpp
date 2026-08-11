// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <span>
#include <string_view>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim::kda_factory_detail {

tt::tt_metal::ComputeConfigDescriptor kda_compute_cfg(
    tt::ARCH arch, const DeviceComputeKernelConfig& config, bool honor_caller_config = true);

void check_allocated_device_tensor(const Tensor& tensor, std::string_view operation_name, std::string_view tensor_name);
void check_layout(
    const Tensor& tensor,
    tt::tt_metal::Layout required_layout,
    std::string_view operation_name,
    std::string_view tensor_name);
void check_dtype(
    const Tensor& tensor,
    tt::tt_metal::DataType required_dtype,
    std::string_view operation_name,
    std::string_view tensor_name);
void check_dtype_in(
    const Tensor& tensor,
    std::span<const tt::tt_metal::DataType> accepted_dtypes,
    std::string_view accepted_dtype_names,
    std::string_view operation_name,
    std::string_view tensor_name);
void check_matching_dtype(
    const Tensor& lhs, const Tensor& rhs, std::string_view operation_name, std::string_view tensor_group_name);
void check_same_device(
    const Tensor& reference, const Tensor& candidate, std::string_view operation_name, std::string_view candidate_name);
void check_interleaved(const Tensor& tensor, std::string_view operation_name, std::string_view tensor_name);
void check_output_interleaved(const tt::tt_metal::MemoryConfig& memory_config, std::string_view operation_name);
void check_compute_config(const DeviceComputeKernelConfig& config, std::string_view operation_name);

struct KdaPrepWorkDist {
    std::vector<tt::tt_metal::CoreCoord> cores;
    std::vector<uint32_t> wi_start;
    std::vector<uint32_t> wi_count;
    tt::tt_metal::CoreRangeSet core_set;
};

KdaPrepWorkDist distribute_prep(tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t core_cap);

}  // namespace ttnn::experimental::prim::kda_factory_detail

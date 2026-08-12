// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_factory_utils.hpp"

#include <algorithm>
#include <enchantum/enchantum.hpp>
#include <set>
#include <tuple>

#include <tt-metalium/work_split.hpp>
#include <tt_stl/assert.hpp>

namespace ttnn::experimental::prim::kda_factory_detail {

void check_allocated_device_tensor(
    const Tensor& tensor, std::string_view operation_name, std::string_view tensor_name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "{}: {} must be an allocated device tensor",
        operation_name,
        tensor_name);
}

void check_layout(
    const Tensor& tensor,
    tt::tt_metal::Layout required_layout,
    std::string_view operation_name,
    std::string_view tensor_name) {
    TT_FATAL(
        tensor.layout() == required_layout,
        "{}: {} must use {} layout, got {}",
        operation_name,
        tensor_name,
        enchantum::to_string(required_layout),
        tensor.layout());
}

void check_dtype(
    const Tensor& tensor,
    tt::tt_metal::DataType required_dtype,
    std::string_view operation_name,
    std::string_view tensor_name) {
    TT_FATAL(
        tensor.dtype() == required_dtype,
        "{}: {} must be {}, got {}",
        operation_name,
        tensor_name,
        enchantum::to_string(required_dtype),
        tensor.dtype());
}

void check_dtype_in(
    const Tensor& tensor,
    std::span<const tt::tt_metal::DataType> accepted_dtypes,
    std::string_view accepted_dtype_names,
    std::string_view operation_name,
    std::string_view tensor_name) {
    TT_FATAL(!accepted_dtypes.empty(), "{}: accepted dtype set must not be empty", operation_name);
    TT_FATAL(
        std::find(accepted_dtypes.begin(), accepted_dtypes.end(), tensor.dtype()) != accepted_dtypes.end(),
        "{}: {} must be {}, got {}",
        operation_name,
        tensor_name,
        accepted_dtype_names,
        tensor.dtype());
}

void check_matching_dtype(
    const Tensor& lhs, const Tensor& rhs, std::string_view operation_name, std::string_view tensor_group_name) {
    TT_FATAL(
        lhs.dtype() == rhs.dtype(),
        "{}: {} must have matching dtypes, got {} and {}",
        operation_name,
        tensor_group_name,
        lhs.dtype(),
        rhs.dtype());
}

void check_same_device(
    const Tensor& reference,
    const Tensor& candidate,
    std::string_view operation_name,
    std::string_view candidate_name) {
    TT_FATAL(
        candidate.device() == reference.device(),
        "{}: {} must be on the same device as the other inputs",
        operation_name,
        candidate_name);
}

void check_interleaved(const Tensor& tensor, std::string_view operation_name, std::string_view tensor_name) {
    TT_FATAL(!tensor.is_sharded(), "{}: {} must use interleaved memory", operation_name, tensor_name);
}

void check_output_interleaved(const tt::tt_metal::MemoryConfig& memory_config, std::string_view operation_name) {
    TT_FATAL(
        !memory_config.is_sharded(),
        "{}: output memory layout must be INTERLEAVED, got {}",
        operation_name,
        enchantum::to_string(memory_config.memory_layout()));
}

void check_compute_config(const DeviceComputeKernelConfig& config, std::string_view operation_name) {
    TT_FATAL(
        !config.packer_l1_acc,
        "{}: packer_l1_acc=true is unsupported because the compute kernel does not accumulate through L1",
        operation_name);
}

tt::tt_metal::ComputeConfigDescriptor kda_compute_cfg(
    tt::ARCH arch, const DeviceComputeKernelConfig& config, bool honor_caller_config) {
    if (!honor_caller_config) {
        return tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false};
    }
    const auto args = get_compute_kernel_config_args(arch, config);
    return tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = std::get<0>(args),
        .fp32_dest_acc_en = std::get<2>(args),
        .dst_full_sync_en = std::get<4>(args),
        .math_approx_mode = std::get<1>(args)};
}

KdaPrepWorkDist distribute_prep(tt::tt_metal::CoreCoord grid, uint32_t total, uint32_t core_cap) {
    const uint32_t max_cores = std::min<uint32_t>(grid.x * grid.y, core_cap);
    const uint32_t count = std::min(total, max_cores);
    TT_FATAL(count > 0, "KDA work distribution needs at least one item (total={})", total);
    const uint32_t base = total / count;
    const uint32_t remainder = total % count;

    KdaPrepWorkDist distribution;
    distribution.cores.reserve(count);
    distribution.wi_start.reserve(count);
    distribution.wi_count.reserve(count);
    uint32_t offset = 0;
    for (uint32_t index = 0; index < count; ++index) {
        const tt::tt_metal::CoreCoord core{index % grid.x, index / grid.x};
        const uint32_t item_count = base + (index < remainder ? 1u : 0u);
        distribution.cores.push_back(core);
        distribution.wi_start.push_back(offset);
        distribution.wi_count.push_back(item_count);
        offset += item_count;
    }
    distribution.core_set = tt::tt_metal::num_cores_to_corerangeset(count, grid, true);
    return distribution;
}

}  // namespace ttnn::experimental::prim::kda_factory_detail

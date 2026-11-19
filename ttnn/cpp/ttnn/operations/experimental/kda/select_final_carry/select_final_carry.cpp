// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry.hpp"

#include <array>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/select_final_carry_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::kda {

namespace {

std::array<uint32_t, 2> semaphore_addresses(const std::vector<tt::tt_metal::GlobalSemaphore>& semaphores) {
    TT_FATAL(semaphores.size() == 2, "select_final_carry: semaphores must be [barrier, arrival]");
    return {static_cast<uint32_t>(semaphores[0].address()), static_cast<uint32_t>(semaphores[1].address())};
}

}  // namespace

ttnn::Tensor select_final_carry(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    uint32_t num_links,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const std::vector<tt::tt_metal::GlobalSemaphore>& semaphores,
    const std::optional<ttnn::Tensor>& actual_end,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    uint32_t sequence_parallel_axis) {
    // Validation runs on cache misses only; the device must be known before then to pick the links.
    TT_FATAL(
        prefix_final.storage_type() == StorageType::DEVICE && prefix_final.is_allocated(),
        "select_final_carry: prefix_final must be an allocated device tensor");
    auto* mesh = prefix_final.device();
    TT_FATAL(
        sequence_parallel_axis < mesh->shape().dims(),
        "select_final_carry: sequence_parallel_axis {} is out of range for a {}-D mesh",
        sequence_parallel_axis,
        mesh->shape().dims());
    const auto topology = ::ttnn::ccl::convert_2d_to_1d_topology(::ttnn::ccl::get_usable_topology(
        prefix_final, tt::tt_fabric::get_fabric_topology(*mesh), sequence_parallel_axis));
    return ttnn::experimental::prim::select_final_carry(
        rank_final,
        prefix_final,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        num_links,
        topology,
        cores,
        semaphore_addresses(semaphores));
}

}  // namespace ttnn::experimental::kda

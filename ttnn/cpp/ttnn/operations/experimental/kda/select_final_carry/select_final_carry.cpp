// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "select_final_carry.hpp"

#include <array>

#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "device/select_final_carry_device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"

namespace ttnn::experimental::kda {

namespace {

struct LineSetup {
    uint32_t num_links;
    tt::tt_fabric::Topology topology;
};

// Validation runs on cache misses only; the device must be known before then to pick the links.
LineSetup line_setup(
    const ttnn::Tensor& prefix_final, uint32_t sequence_parallel_axis, std::optional<uint32_t> num_links) {
    TT_FATAL(
        prefix_final.storage_type() == StorageType::DEVICE && prefix_final.buffer() != nullptr,
        "select_final_carry: prefix_final must be an allocated device tensor");
    auto* mesh = prefix_final.device();
    TT_FATAL(
        sequence_parallel_axis < mesh->shape().dims(),
        "select_final_carry: sequence_parallel_axis {} is out of range for a {}-D mesh",
        sequence_parallel_axis,
        mesh->shape().dims());
    const auto topology = ::ttnn::ccl::convert_2d_to_1d_topology(::ttnn::ccl::get_usable_topology(
        prefix_final, tt::tt_fabric::get_fabric_topology(*mesh), sequence_parallel_axis));
    const auto available_links =
        static_cast<uint32_t>(ttnn::operations::ccl::common::get_num_links(*mesh, sequence_parallel_axis));
    TT_FATAL(
        num_links.value_or(available_links) <= available_links,
        "select_final_carry: num_links {} exceeds the {} links of the sequence-parallel axis",
        num_links.value_or(available_links),
        available_links);
    return {num_links.value_or(available_links), topology};
}

std::optional<std::array<uint32_t, 2>> semaphore_addresses(
    const std::optional<std::vector<tt::tt_metal::GlobalSemaphore>>& semaphores) {
    if (!semaphores) {
        return std::nullopt;
    }
    TT_FATAL(semaphores->size() == 2, "select_final_carry: semaphores must be [barrier, arrival]");
    return std::array<uint32_t, 2>{
        static_cast<uint32_t>((*semaphores)[0].address()), static_cast<uint32_t>((*semaphores)[1].address())};
}

}  // namespace

ttnn::Tensor select_final_carry(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::optional<ttnn::Tensor>& actual_end,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    uint32_t sequence_parallel_axis,
    std::optional<uint32_t> num_links,
    const std::optional<std::vector<tt::tt_metal::CoreCoord>>& cores,
    bool prefix_on_last_rank,
    const std::optional<std::vector<tt::tt_metal::GlobalSemaphore>>& semaphores) {
    const auto line = line_setup(prefix_final, sequence_parallel_axis, num_links);
    return ttnn::experimental::prim::select_final_carry(
        rank_final,
        prefix_final,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        line.num_links,
        line.topology,
        prefix_on_last_rank,
        cores,
        semaphore_addresses(semaphores))[0];
}

std::tuple<ttnn::Tensor, ttnn::Tensor> select_final_carry_with_projection(
    const ttnn::Tensor& rank_final,
    const ttnn::Tensor& prefix_final,
    const ttnn::Tensor& projection_input,
    const ttnn::Tensor& projection_weight,
    const ttnn::Tensor& actual_start,
    uint32_t local_rows,
    const std::vector<tt::tt_metal::CoreCoord>& cores,
    const ttnn::operations::matmul::MatmulMultiCoreReuseMultiCastProgramConfig& projection_program_config,
    const std::optional<ttnn::Tensor>& actual_end,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    uint32_t sequence_parallel_axis,
    std::optional<uint32_t> num_links,
    bool prefix_on_last_rank,
    const std::optional<ttnn::MemoryConfig>& projection_memory_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& projection_compute_kernel_config,
    const std::optional<std::vector<tt::tt_metal::GlobalSemaphore>>& semaphores) {
    const auto line = line_setup(prefix_final, sequence_parallel_axis, num_links);
    const auto outputs = ttnn::experimental::prim::select_final_carry(
        rank_final,
        prefix_final,
        memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
        actual_start,
        actual_end,
        sequence_parallel_axis,
        local_rows,
        line.num_links,
        line.topology,
        prefix_on_last_rank,
        cores,
        semaphore_addresses(semaphores),
        ttnn::experimental::prim::FinalCarryProjectionParams{
            .program_config = projection_program_config,
            // HiFi2 unless given, approximation off, FP32 accumulation.
            .compute_kernel_config = ttnn::init_device_compute_kernel_config(
                prefix_final.device()->arch(),
                projection_compute_kernel_config,
                tt::tt_metal::MathFidelity::HiFi2,
                /*default_approx_mode=*/false,
                /*default_fp32_acc=*/true),
            .output_mem_config = projection_memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG)},
        projection_input,
        projection_weight);
    return {outputs[0], outputs[1]};
}

}  // namespace ttnn::experimental::kda
